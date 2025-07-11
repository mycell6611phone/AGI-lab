"""
agent_personas.py

Purpose:
Defines and manages internal sub-agents or "personas" with different reasoning styles or expertise.
"""

class Persona:
    """A simple agent persona with a name and reasoning style."""
    def __init__(self, name, style, reason_fn):
        self.name = name
        self.style = style
        self.reason = reason_fn  # function(input, context) -> str

class AgentPersonas:
    """Handles the creation and coordination of multiple sub-agents or personas."""

    def __init__(self):
        self.personas = {}
        print("[AgentPersonas] Initialized")

    def add_persona(self, persona):
        self.personas[persona.name] = persona
        print(f"[AgentPersonas] Added persona: {persona.name}")

    def get_persona(self, name):
        return self.personas.get(name)

    def list_personas(self):
        return list(self.personas.values())

    def remove_persona(self, name):
        if name in self.personas:
            del self.personas[name]
            print(f"[AgentPersonas] Removed persona: {name}")

# === Example persona reasoning functions ===

def critic_reason(input_text, context=None):
    return f"[Critic] My main concern is: could this be wrong? Here's a critique of the input: '{input_text}'."

def optimist_reason(input_text, context=None):
    return f"[Optimist] I see the positive side! Here's what could go right with: '{input_text}'."

def contrarian_reason(input_text, context=None):
    return f"[Contrarian] Let's consider the opposite view of: '{input_text}'."

# === Example usage: register personas ===
if __name__ == "__main__":
    ap = AgentPersonas()
    ap.add_persona(Persona("Critic", "skeptical", critic_reason))
    ap.add_persona(Persona("Optimist", "positive", optimist_reason))
    ap.add_persona(Persona("Contrarian", "contrarian", contrarian_reason))
    for p in ap.list_personas():
        print(f"{p.name}: {p.style}, {p.reason('AGI will take over the world.')}")

