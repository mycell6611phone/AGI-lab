"""
coreloop.py
~~~~~~~~~~~

Main AGI mind loop scaffold.
Stitches together modular subsystems into a continuous cognitive loop.

* Each cognitive step is documented.
* All logic is now correctly inside the main loop.
* Ready for future expansion/experimentation.

Dependencies:
    - memory.py
    - goal_manager.py
    - planner.py
    - agent_personas.py
    - self_critic.py
    - emotion.py
    - experimenter.py
    - trainer.py
    - interface.py
    - prompts.py
    - config.py
    - llm_client.py
    - tool_manager.py
"""

from __future__ import annotations
import argparse
import logging

# Import all major subsystems (should each be independently testable)
import memory  # Handles episodic/semantic memory storage and recall
import goal_manager  # Tracks agent goals
import planner  # Generates candidate actions/plans
import agent_personas  # Houses multiple “sub-agents” (reasoning styles/roles)
import self_critic  # Self-critique and output analysis
import emotion  # Models current mood/motivation
import experimenter  # Runs “what if” tests, records discoveries
import trainer  # Handles self-improvement, training, fine-tuning
import interface  # User/environment I/O abstraction
from prompts import get_prompt, WEBTOOL_SYSTEM_PROMPT  # Loads phase/system prompts
from config import settings  # Loads environment config/settings
from llm_client import call_llm  # Calls LLM for thought generation



# ---- TOOL REGISTRATION ----
from tool_manager import register_tool
from webtool import google_search

register_tool("google_search", google_search)

 # Make sure this exists in memory.py

# Add more tools here as you develop them:
# register_tool("wiki_search", wiki_search)
# register_tool("run_code", run_code)

logger = logging.getLogger(__name__)

class AGIMindLoop:
    """Main cognitive loop for the agent."""

    def __init__(self) -> None:
        """Initialize subsystem handles for the AGI."""
        # VectorStore: fast vector search for semantic memory (FAISS backend)
        self.vector_store = memory.VectorStore(settings.faiss_path)
        # MetaStore: structured memory (SQLite backend)
        self.meta_store = memory.MetaStore(settings.sqlite_path)
        # Interface: manages input/output, decoupling IO from logic
        self.iface = interface.Interface()

    def run(self, cycles: int = 0) -> None:
        """
        Main mind loop:
        If cycles == 0, runs forever (true AGI mode).
        Else, runs the specified number of cycles for test/debug.
        """
        if cycles == 0:
            cycle = 3
            while True:
                logger.info("Starting cycle %s", cycle)
                self.cognitive_cycle(cycle)
                cycle += 1
        else:
            for cycle in range(1, cycles + 1):
                logger.info("Starting cycle %s", cycle)
                self.cognitive_cycle(cycle)

    def cognitive_cycle(self, cycle_num: int):
        """
        One full AGI mind cycle.
        Each step is a key part of the cognitive architecture.
        """
        # -----------------------------------------------------------------
        # 1. Perceive / Input
        # -----------------------------------------------------------------
        #print("[Perceive]  # Step 1: Agent perceives the world (user, environment, self)")
        perceived_input = self.iface.get_input()
        print(f"[Perceive] Received: {perceived_input}")

        # -----------------------------------------------------------------
        # 2. Recall
        # -----------------------------------------------------------------
        #print("[Recall]  # Step 2: Retrieve relevant memories (episodic/semantic/goals)")
        memories = memory.recall(
            perceived_input,
            k_embeddings=5,
            vector_store=self.vector_store,
            meta_store=self.meta_store,
        )
        print(f"[Recall] Retrieved memories: {memories}")

        # -----------------------------------------------------------------
        # 3. Think / Plan
        # -----------------------------------------------------------------
       #print("[Think/Plan]  # Step 3: Generate candidate actions/thoughts based on input + memory")
        system_prompt = get_prompt("plan") + "\n" + WEBTOOL_SYSTEM_PROMPT
        plan = call_llm(perceived_input, system_msg=system_prompt)
        print(f"[Think/Plan] Plan output: {plan}")

        # -----------------------------------------------------------------
        # TOOL CALL HANDLING (after plan)
        # -----------------------------------------------------------------
        from tool_manager import extract_tool_calls, execute_tool_call
        tool_calls = extract_tool_calls(plan)
        tool_results = []
        if tool_calls:
            #print(f"[ToolHandler] Tool calls detected: {tool_calls}")
            for name, args in tool_calls:
                result = execute_tool_call(name, args)
                tool_results.append((name, args, result))
                print(f"[ToolHandler] {name}({args}) → {result}")
            # Compose plan_context with tool results for next phase
            plan_context = plan + "\n\n" + "\n".join(
                f"[TOOL RESULT] {name}({args}) → {result}" for name, args, result in tool_results
            )
        else:
            plan_context = plan

        
        # 4. Debate/Filter Tool Results (accept/reject, candidate memory)
        #print("[Debate]  # Step 4: Debate and filter raw tool/web data")

        debate_prompt = (
            "Below is information returned by a tool call (e.g., web search). "
            "1. Propose what, if anything, should be saved to long-term memory or used for the current task as:\n"
         "   CANDIDATE_MEMORY: \"<your summary or most relevant data>\"\n"
         "2. Debate: DECISION: [ACCEPT|REJECT]\n"
         "   JUSTIFICATION: <Explain why this info is useful, trustworthy, or not.>\n"
         "If nothing should be remembered, say REJECT."
        )

        debate_input = (
            f"User input: {perceived_input}\n"
            f"Tool results: {tool_results}\n"
            f"Context: {plan_context}\n"
        )

        messages = [
            {"role": "system", "content": debate_prompt},
            {"role": "user", "content": debate_input}
        ]

        # LLM “A”
        debate_a = call_llm(messages, system_msg=debate_prompt)
        # LLM “B” (can swap persona, model, or seed if desired)
        debate_b = call_llm(messages, system_msg=debate_prompt)

        def parse_debate_response(text):
            import re
            mem = ""
            dec = ""
            just = ""
            mem_match = re.search(r'CANDIDATE_MEMORY: "?(.+?)"?(\n|$)', text, re.DOTALL)
            if mem_match:
                mem = mem_match.group(1).strip()
            dec_match = re.search(r'DECISION:\s*\[?(ACCEPT|REJECT)\]?', text)
            if dec_match:
                dec = dec_match.group(1)
            just_match = re.search(r'JUSTIFICATION:\s*(.*)', text)
            if just_match:
                just = just_match.group(1).strip()
            return mem, dec, just

        mem_a, dec_a, just_a = parse_debate_response(debate_a)
        mem_b, dec_b, just_b = parse_debate_response(debate_b)

        

        # Decide what to pass to memory/execution:
        if dec_a == "ACCEPT" and dec_b == "ACCEPT":
            accepted_memory = mem_a or mem_b  # Or synthesize/merge if you want
            print(f"[Debate] Memory accepted: {accepted_memory}")
        else:
            accepted_memory = None
            

        
        # -----------------------------------------------------------------
        # 4. Critique
        # -----------------------------------------------------------------
        #print("[Critique]  # Step 4: Critically analyze plan (self-reflection, cross-critique, thought experiments)")
        phase_prompt = get_prompt("critique")
        critique = call_llm(plan_context, system_msg=phase_prompt)
        print(f"[Critique] Output: {critique}")

        # -----------------------------------------------------------------
        # 5. Decide / Act
        # -----------------------------------------------------------------
        print("[Decide/Act]  # Step 5: Choose best plan/action, possibly after agent debate")
        from agent_personas import Persona, critic_reason, optimist_reason, contrarian_reason
        personas = agent_personas.AgentPersonas()
        personas.add_persona(Persona("Critic", "skeptical", critic_reason))
        personas.add_persona(Persona("Optimist", "positive", optimist_reason))
        personas.add_persona(Persona("Contrarian", "contrarian", contrarian_reason))
        decision = personas.list_personas()
        if decision:
            #print("[Decide/Act] Persona responses:")
            for persona in decision:
                output = persona.reason(plan_context)   # Use plan_context with tool results
                print(f"  [{persona.name}] {output}")
        else:
            print("[Decide/Act] No personas available.")
        #print(f"[Decide/Act] Personas output: {[p.name for p in decision]}")

        # -----------------------------------------------------------------
        # 6. Explain
        # -----------------------------------------------------------------
        #print("[Explain]  # Step 6: Generate a self-explanation for transparency and debug")
        explanation_prompt = (
            f"User input: {perceived_input}\n"
            f"Plan: {plan_context}\n"
            f"Critique: {critique}\n"
            "Explain your plan and critique to the user in clear, plain language."
        )
        explanation = call_llm(
            explanation_prompt,
            system_msg="You are an AGI explaining its reasoning to a human.",
        )
        self.iface.send_output(explanation)
        print(f"[Explain] Output: {explanation}")

        # -----------------------------------------------------------------
        # 7. Execute
        # -----------------------------------------------------------------
        #print("[Execute]  # Step 7: Take action, run an experiment, or output a result")
        exp = experimenter.Experimenter()
        execution_result = exp.run_experiment("test hypothesis")
        #print(f"[Execute] Experiment result: {execution_result}")

        # -----------------------------------------------------------------
        # 8. Reflect
        # -----------------------------------------------------------------
        print("[Reflect]  # Step 8: Assess outcome, mood, feedback, drift, surprise")
        emo = emotion.EmotionEngine()
        current_mood = emo.get_mood()
        #print(f"[Reflect] Current mood: {current_mood}")

        # -----------------------------------------------------------------
        # 9. Remember
        # -----------------------------------------------------------------
       # print("[Remember]  # Step 9: Store new episodic memory, update beliefs/skills, tag importance/novelty")
        new_memory = {
            "cycle": cycle_num,
            "input": perceived_input,
            "plan": plan_context,
            "critique": critique,
            "execution_result": execution_result,
            "mood": current_mood,
        }
        # Save memory using memory API (TODO: expand for full tagging/embedding)
        print(f"[Remember] Stored new memory: {new_memory}")

        # -----------------------------------------------------------------
        # 10. Self-Improve
        # -----------------------------------------------------------------
        #print("[Self-Improve]  # Step 10: Propose code/prompt/memory filter changes for continual improvement")
        tr = trainer
        improvement = tr.schedule_training([], "llama3")  # Placeholder: train if enough high-quality memory
        print(f"[Self-Improve] Improvement result: {improvement}")

        # --- End of cycle ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AGI mind loop scaffold")
    parser.add_argument("--cycles", type=int, default=0, help="number of cycles to run (0 for infinite)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    loop = AGIMindLoop()
    loop.run(cycles=args.cycles)

