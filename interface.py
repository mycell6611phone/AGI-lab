"""
interface.py

Purpose:
Handles all external input/output for the AGI system, including user interactions, environment signals, and logging.
"""

class Interface:
    """Manages communication between the AGI and the external world."""

    def __init__(self):
        """Initialize input/output channels."""
        pass

    def get_input(self, end_marker="END", comment_lines=False):
        """
        Obtain multi-line input from user.
        If comment_lines=True, each line is commented out (for code blocks).
        Input ends when the user enters the end_marker.
        """
        prompt_msg = f"Enter your input (multi-line supported). Finish with '{end_marker}' on a line by itself:"
        if comment_lines:
            prompt_msg += " (Each line will be commented out with '#')"
        print(prompt_msg)
        lines = []
        while True:
            line = input()
            if line.strip() == end_marker:
                break
            if comment_lines:
                lines.append(f"# {line}")
            else:
                lines.append(line)
        block = "\n".join(lines)
        return block

    def send_output(self, output):
        """
        Send output or response to user/environment.
        For now, prints to the terminal.
        """
        print(f"AGI says: {output}")

    def log_event(self, event):
        """
        Log important events, messages, or errors.
        For now, prints to the terminal prefixed with LOG:.
        """
        print(f"LOG: {event}")

