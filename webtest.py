import requests
import json
import time

# ---- CONFIGURATION ----

LLAMA_URL = "http://localhost:11434/api/chat"  # Adjust for your Ollama/Llama API

# ---- GOOGLE SEARCH TOOL ----
def google_search(query, num_results=2):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "num": num_results,
    }
    resp = requests.get(url, params=params)
    if not resp.ok:
        return f"Google Search failed: {resp.status_code} {resp.text}"
    data = resp.json()
    results = []
    for item in data.get("items", []):
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        results.append(f"{title}\n{snippet}\n{link}")
    return "\n\n".join(results) if results else "No results found."


# ---- LLAMA CALLER ----
def call_llama3(messages, model="llama3"):
    """
    Call your local Llama-3 model running on localhost (Ollama or similar).
    messages: List of {"role": "user"/"system"/"assistant", "content": "..."}
    """
    data = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    resp = requests.post(LLAMA_URL, json=data)
    resp.raise_for_status()
    result = resp.json()
    print("[LLAMA RAW RESULT]:", result)   
    return result['message']['content'].strip()


# ---- AGENT LOOP ----
def agent_web_debate(query):
    log_accepted = []
    log_rejected = []

    # SYSTEM PROMPT: Tell LLM about the web_search tool and the debate format
    system_prompt = (
        "You are an autonomous AI. You can use the tool: "
        "CALL: google_search(\"your query\") to fetch real-time information from Google. "
        "After receiving results, read and propose what should be saved in memory as: "
        "CANDIDATE_MEMORY: \"<your summary>\".\n"
        "You will then participate in a debate: DECISION: [ACCEPT|REJECT]\n"
        "JUSTIFICATION: <why should this info be kept or not?>"
    )

    # 1. First LLM: Formulate tool call (simulate user question)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {query}"}
    ]
    llm_out = call_llama3(messages)
    print(f"\n[LLM1 Output]: {llm_out}")

    # 2. If LLM calls tool, extract query and perform search
    if "CALL: google_search" in llm_out:
        search_query = llm_out.split("google_search(", 1)[1].split(")")[0].strip('"')
        search_result = google_search(search_query)
        print(f"\n[Google Search Results]:\n{search_result}\n")

        messages.append({"role": "assistant", "content": f"[Google Search Results]:\n{search_result}"})

        # 3. First LLM proposes candidate memory
        llm1_mem = call_llama3(messages)
        print(f"\n[LLM1 Candidate Memory]: {llm1_mem}")

        # 4. Start debate: Both LLMs evaluate the candidate
        debate_prompt = (
            f"The following information is proposed for memory:\n"
            f"{llm1_mem}\n"
            f"Debate: Should this be stored in long-term memory?\n"
            f"Reply as:\nDECISION: [ACCEPT|REJECT]\nJUSTIFICATION: ..."
        )
        # Two LLMs independently judge (could use two models, or two calls)
        debate_msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": debate_prompt}
        ]
        decision_a = call_llama3(debate_msg)
        decision_b = call_llama3(debate_msg)
        print(f"\n[LLM-A Debate]: {decision_a}")
        print(f"[LLM-B Debate]: {decision_b}")

        # 5. Parse decisions and log accordingly
        accepted = "ACCEPT" in decision_a and "ACCEPT" in decision_b
        debate_log = {
            "query": query,
            "search_query": search_query,
            "search_result": search_result,
            "candidate_memory": llm1_mem,
            "debate_a": decision_a,
            "debate_b": decision_b,
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        }

        if accepted:
            log_accepted.append(debate_log)
            print("\n[INFO SAVED TO ACCEPTED LOG]\n")
        else:
            log_rejected.append(debate_log)
            print("\n[INFO SAVED TO REJECTED LOG]\n")

        # 6. Save logs
        with open("accepted_info_log.json", "w") as fa:
            json.dump(log_accepted, fa, indent=2)
        with open("rejected_info_log.json", "w") as fr:
            json.dump(log_rejected, fr, indent=2)
        print("[Logs updated.]")

    else:
        print("[LLM did not call the web tool. No search performed.]")

if __name__ == "__main__":
    agent_web_debate("What is the latest version of Python?")
