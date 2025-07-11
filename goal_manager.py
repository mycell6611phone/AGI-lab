"""
goal_manager.py

Purpose:
Manages agent goals—adds, parses, prioritizes, tracks subgoals and metadata, and persists goals across runs.

This module provides the GoalManager class, which is responsible for:
- Parsing user input to extract goals, subgoals, and metadata (like priority)
- Adding, completing, and listing goals and subgoals
- Marking goals/subgoals as complete
- Saving and loading persistent goals from disk
- Displaying goals and progress in a human-readable way
"""

import re
import time
import json
import os

GOAL_SAVE_PATH = "goals.json"  # Path to persist goal list between runs

class GoalManager:
    """
    Tracks and manages AGI agent goals, subgoals, and metadata.

    Main features:
    - Parses new goals and subgoals from input
    - Stores metadata such as priorities, creation times, and goal IDs
    - Supports marking goals and individual subgoals as complete
    - Saves and loads goals to/from disk
    - Presents readable status for all current and completed goals
    """

    def __init__(self):
        """
        Initialize the GoalManager.
        Loads saved goals from disk (if any).
        Sets up in-memory goal list and next goal ID.
        """
        self.goals = []    # List of goal dicts (see add_goal for schema)
        self.next_id = 1   # Incrementing integer ID for new goals
        print("[GoalManager] Initialized")
        self.load_goals()  # Populate from file if possible

    def parse_goal(self, user_input):
        """
        Parse a raw user string into a main goal, list of subgoals, and metadata dict.

        Supports lines like:
            Goal: Learn Python [priority=3]
            Subgoal: Install Python
            Subgoal: Write hello world

        Args:
            user_input (str): Freeform, potentially multiline user description of goals.

        Returns:
            Tuple[str, List[str], dict]:
                - main_goal (str or None)
                - subgoals (list of strings)
                - metadata (dict, e.g. {'priority': 3})
        """
        lines = user_input.splitlines()
        current_goal = None
        subgoals = []
        metadata = {}
        for line in lines:
            # Main goal, potentially with metadata in [key=val] brackets
            if line.lower().startswith("goal:"):
                goal_text = line.split("goal:",1)[1].strip()
                # Extract [priority=3,tag=x] style metadata from brackets
                meta = re.findall(r"\[(.*?)\]", goal_text)
                if meta:
                    for m in meta:
                        for item in m.split(","):
                            k, v = item.split("=",1)
                            metadata[k.strip()] = v.strip()
                    goal_text = re.sub(r"\[.*?\]", "", goal_text).strip()
                current_goal = goal_text
            # Each subgoal is a line beginning with subgoal:
            elif line.lower().startswith("subgoal:"):
                subgoal_text = line.split("subgoal:",1)[1].strip()
                subgoals.append(subgoal_text)
        return current_goal, subgoals, metadata

    def add_goal(self, goal, subgoals=None, metadata=None):
        """
        Adds a new goal (and its subgoals and metadata) to the list.

        Args:
            goal (str): The main goal string.
            subgoals (list[str]): List of subgoal strings (optional).
            metadata (dict): Dictionary of metadata, e.g., {'priority': 3} (optional).
        """
        gid = self.next_id
        self.next_id += 1
        # Each subgoal is a dict: {'text': str, 'done': False}
        subgoal_objs = [{"text": s, "done": False} for s in (subgoals or [])]
        gobj = {
            "id": gid,
            "goal": goal,
            "subgoals": subgoal_objs,
            "metadata": metadata or {},
            "created": time.strftime("%Y-%m-%d %H:%M"),
            "priority": int((metadata or {}).get("priority", 1)),
            "active": True,  # True if incomplete, False if done
        }
        self.goals.append(gobj)
        print(f"[GoalManager] Added goal #{gid}: {goal}")
        if subgoals:
            for i, s in enumerate(subgoals, 1):
                print(f"    └─ Subgoal {i}: {s}")
        self.save_goals()

    def complete_goal(self, goal_id):
        """
        Mark a goal (and all its subgoals) as completed.

        Args:
            goal_id (int): The numeric ID of the goal to complete.
        """
        for g in self.goals:
            if int(g['id']) == int(goal_id):
                g['active'] = False
                # Mark all subgoals as done
                for sg in g['subgoals']:
                    sg['done'] = True
                print(f"[GoalManager] Marked goal #{goal_id} as DONE (and all subgoals).")
                self.save_goals()
                return
        print(f"[GoalManager] Goal #{goal_id} not found.")

    def complete_subgoal(self, goal_id, subgoal_idx):
        """
        Mark a single subgoal (by index) as complete.
        If all subgoals are done, also mark the goal as done.

        Args:
            goal_id (int): ID of the parent goal.
            subgoal_idx (int): 1-based index of the subgoal to mark complete.
        """
        for g in self.goals:
            if int(g['id']) == int(goal_id):
                if 1 <= subgoal_idx <= len(g['subgoals']):
                    g['subgoals'][subgoal_idx-1]['done'] = True
                    print(f"[GoalManager] Marked subgoal {subgoal_idx} of goal #{goal_id} as DONE.")
                    # If all subgoals now complete, also mark goal done
                    if all(sg['done'] for sg in g['subgoals']):
                        g['active'] = False
                        print(f"[GoalManager] All subgoals complete. Marked goal #{goal_id} as DONE.")
                    self.save_goals()
                    return
                else:
                    print(f"[GoalManager] Subgoal {subgoal_idx} not found in goal #{goal_id}.")
                    return
        print(f"[GoalManager] Goal #{goal_id} not found.")

    def update(self, user_input, context, plan, status):
        """
        Unified handler for new goals and completion commands.

        If user_input starts with 'complete:', parses it as a command
        (supports 'complete: <goal_id>' and 'complete: <goal_id>.<subgoal_idx>').
        Otherwise, tries to parse user_input as a new goal + subgoals.

        Args:
            user_input (str): User command or goal text.
            context, plan, status: (Unused, reserved for future context).
        """
        print("[GoalManager] update called with:")
        print("  user_input:", user_input)
        # If user wants to mark goal/subgoal as complete, do it and list
        if user_input.lower().startswith("complete:"):
            try:
                parts = user_input.split(":",1)[1].strip().split(".")
                goal_id = int(parts[0])
                if len(parts) == 2:
                    subgoal_idx = int(parts[1])
                    self.complete_subgoal(goal_id, subgoal_idx)
                else:
                    self.complete_goal(goal_id)
            except Exception as e:
                print("[GoalManager] Could not parse complete command:", e)
            self.list_goals()
            return  # Don't add a new goal if it was a completion command

        # Otherwise, treat input as a new goal definition
        main_goal, subgoals, metadata = self.parse_goal(user_input)
        if main_goal:
            self.add_goal(main_goal, subgoals=subgoals, metadata=metadata)
        self.list_goals()

    def list_goals(self):
        """
        Print a summary of all goals, sorted by priority.

        Shows status, all subgoals, progress bar, and metadata if present.
        """
        print("[GoalManager] Listing all goals")
        # Higher priority numbers sort first
        for g in sorted(self.goals, key=lambda g: -g['priority']):
            status = "ACTIVE" if g["active"] else "DONE"
            print(f"  [#{g['id']}] {g['goal']} (Priority: {g['priority']}, {status})")
            if g["subgoals"]:
                for idx, sg in enumerate(g["subgoals"], 1):
                    done_str = "✓" if sg.get("done") else " "
                    print(f"      └─ Subgoal {idx}: [{done_str}] {sg['text']}")
            if g["metadata"]:
                print(f"      Metadata: {g['metadata']}")
            # Subgoal progress: how many are complete
            if g["subgoals"]:
                total = len(g["subgoals"])
                done = sum(1 for sg in g["subgoals"] if sg.get("done"))
                print(f"      Progress: {done}/{total} subgoals complete")

    def save_goals(self):
        """
        Write the goal list to disk (as JSON), so it persists across program runs.
        """
        try:
            with open(GOAL_SAVE_PATH, "w") as f:
                json.dump(self.goals, f)
        except Exception as e:
            print(f"[GoalManager] Error saving goals: {e}")

    def load_goals(self):
        """
        Load goals from disk (if present), and upgrade them as needed for compatibility.

        - Ensures all goals have integer IDs
        - Upgrades old subgoal formats to the latest schema
        - Sets next_id correctly so new goals get unique IDs
        """
        if os.path.exists(GOAL_SAVE_PATH):
            try:
                with open(GOAL_SAVE_PATH, "r") as f:
                    self.goals = json.load(f)
                # Upgrade all goals and subgoals to ensure schema consistency
                for g in self.goals:
                    g['id'] = int(g['id'])
                    # Older versions: subgoals may be just a list of strings
                    if g.get('subgoals') and isinstance(g['subgoals'][0], str):
                        g['subgoals'] = [{"text": s, "done": False} for s in g['subgoals']]
                    # Every subgoal must have a 'done' key
                    for sg in g.get('subgoals', []):
                        if 'done' not in sg:
                            sg['done'] = False
                # Set the next available goal ID
                if self.goals:
                    self.next_id = max(g['id'] for g in self.goals) + 1
                else:
                    self.next_id = 1
                print(f"[GoalManager] Loaded {len(self.goals)} goals from disk.")
            except Exception as e:
                print(f"[GoalManager] Error loading goals: {e}")
                self.goals = []
                self.next_id = 1
        else:
            self.goals = []
            self.next_id = 1

