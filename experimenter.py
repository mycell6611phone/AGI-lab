import os
import traceback

class Experimenter:
    def __init__(self):
        print("[Experimenter] Initialized")

    def maybe_run(self, context, user_input, goal_manager=None):
        """
        Loops through all goals/subgoals, executes code for open subgoals,
        and marks them complete if successful.
        """
        print("[Experimenter] Running autonomous subgoal executor...")
        if goal_manager is None:
            print("[Experimenter] No GoalManager provided! Skipping automation.")
            return

        for goal in goal_manager.goals:
            if not goal.get("active"):
                continue
            gid = goal["id"]
            for idx, subg in enumerate(goal["subgoals"], 1):
                if subg.get("done"):
                    continue
                subgoal_text = subg["text"].lower()
                print(f"[Experimenter] Attempting subgoal {gid}.{idx}: {subgoal_text}")

                code = None

                if "create a sample csv file" in subgoal_text:
                    code = (
                        "import csv\n"
                        "def create_sample_csv(filename):\n"
                        "    with open(filename, 'w', newline='') as f:\n"
                        "        writer = csv.writer(f)\n"
                        "        writer.writerow(['col1', 'col2'])\n"
                        "        for i in range(10):\n"
                        "            writer.writerow([f'row{i+1}A', f'row{i+1}B'])\n"
                        "create_sample_csv('test.csv')\n"
                    )

                elif "read the csv file and count the number of rows" in subgoal_text:
                    code = (
                        "import csv\n"
                        "def count_csv_rows(filename):\n"
                        "    with open(filename, 'r') as f:\n"
                        "        reader = csv.reader(f)\n"
                        "        next(reader)  # skip header\n"
                        "        return sum(1 for _ in reader)\n"
                        "row_count = count_csv_rows('test.csv')\n"
                        "print(f'Total rows (not counting header): {row_count}')\n"
                    )

                elif "print the total row count" in subgoal_text:
                    code = (
                        "import csv\n"
                        "def count_csv_rows(filename):\n"
                        "    with open(filename, 'r') as f:\n"
                        "        reader = csv.reader(f)\n"
                        "        next(reader)\n"
                        "        return sum(1 for _ in reader)\n"
                        "row_count = count_csv_rows('test.csv')\n"
                        "print(f'Total rows: {row_count}')\n"
                    )

                elif "test the script end-to-end" in subgoal_text:
                    code = (
                        "import csv\n"
                        "def create_sample_csv(filename):\n"
                        "    with open(filename, 'w', newline='') as f:\n"
                        "        writer = csv.writer(f)\n"
                        "        writer.writerow(['col1', 'col2'])\n"
                        "        for i in range(10):\n"
                        "            writer.writerow([f'row{i+1}A', f'row{i+1}B'])\n"
                        "def count_csv_rows(filename):\n"
                        "    with open(filename, 'r') as f:\n"
                        "        reader = csv.reader(f)\n"
                        "        next(reader)\n"
                        "        return sum(1 for _ in reader)\n"
                        "filename = 'test.csv'\n"
                        "create_sample_csv(filename)\n"
                        "row_count = count_csv_rows(filename)\n"
                        "print(f'Test file: {filename}, rows: {row_count}')\n"
                    )

                # Add more subgoal handlers here as needed!

                if code:
                    print(f"[Experimenter] Executing code for subgoal {gid}.{idx}:\n{code}")
                    try:
                        exec(code, {})
                        print(f"[Experimenter] Subgoal {gid}.{idx} succeeded.")
                        if goal_manager:
                            goal_manager.complete_subgoal(gid, idx)
                    except Exception as e:
                        print(f"[Experimenter] ERROR executing subgoal {gid}.{idx}:", e)
                        print(traceback.format_exc())
                        # Do not mark as complete
                else:
                    print(f"[Experimenter] No code handler for subgoal {gid}.{idx}: {subgoal_text}")

        print("[Experimenter] Finished all subgoal attempts.")

    # === ADDED UNIVERSAL RUN_EXPERIMENT STUB ===
    def run_experiment(self, hypothesis):
        """
        Stub method for compatibility with AGI main loop.
        """
        print(f"[Experimenter] run_experiment called (stub): {hypothesis}")
        return f"Stub: '{hypothesis}' experiment completed."


