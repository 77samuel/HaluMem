import os
import re
import time
import json
import copy
import traceback
from datetime import datetime, timezone
from dotenv import load_dotenv
# Multiprocessing DISABLED in eval_memzero

from tqdm import tqdm
from mem0 import MemoryClient
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed
)

from llms import llm_request
from prompts import PROMPT_MEMZERO


load_dotenv()

# Update custom instructions
custom_instructions = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""

TEMPLATE_MEM0 = """Memories for user {user_id}:

    {memories}
"""

RETRY_TIMES = 10
WAIT_TIME = 60

client = MemoryClient(
    api_key=os.getenv("MEM0_API_KEY"),
)
client.project.update(custom_instructions=custom_instructions)


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_fixed(WAIT_TIME),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True
)
def add_memory(client, user_id, message, timestamp):
    # Compatibility patch for Mem0 async API (2026).
    # Original HaluMem implementation assumed synchronous add() returning
    # {"results": [...]} directly. Mem0's current API returns only
    # {"event_id": ..., "status": "PENDING"} from add(), with actual
    # extracted memories only available via a later get_all() call once
    # background processing completes. This restores the original return
    # contract (a dict with a "results" key shaped like get_all()'s output)
    # without altering any evaluation/scoring logic downstream.
    start = time.time()

    client.add(
        message,
        user_id=user_id,
        version="v2",
        output_format="v1.1",
        timestamp=timestamp,
        enable_graph=False,
        async_mode=False,
    )

    # Poll get_all() until the async extraction finishes or we time out.
    memories_result = {"results": []}
    for _ in range(12):  # up to ~60s
        time.sleep(5)
        try:
            fetched = client.get_all(filters={"AND": [{"user_id": user_id}]})
            if fetched.get("results"):
                memories_result = fetched
                break
        except Exception:
            continue

    duration_ms = (time.time() - start) * 1000

    return memories_result, duration_ms


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_fixed(WAIT_TIME),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True
)
def search_memory(
    client, 
    query, 
    user_id, 
    top_k=20
):

    start = time.time()

    # Compatibility patch for Mem0 API (2026): top-level user_id no longer
    # accepted alongside filters; filters alone is sufficient and required.
    results = client.search(
        query=query,
        top_k=top_k,
        output_format="v1.1",
        version="v2",
        filters={"AND": [{"user_id": f"{user_id}"}]},
    )

    memories = [
        {
            "memory": memory["memory"],
            "timestamp": memory["created_at"],
            "score": round(memory["score"], 2),
        }
        for memory in results["results"]
    ]

    memories = [
        f"{item['timestamp']}: {item['memory']}" for item in memories
    ]

    context = TEMPLATE_MEM0.format(
        user_id=user_id,
        memories=json.dumps(memories, indent=4)
    )

    duration_ms = (time.time() - start) * 1000

    return context, memories, duration_ms


def extract_user_name(persona_info: str):
    match = re.search(r'Name:\s*(.*?); Gender:', persona_info)

    if match:
        username = match.group(1).strip()
        return username
    else:
        raise ValueError("No name found.")
    

def process_user(user_data, top_k, save_path):

    user_name = extract_user_name(user_data["persona_info"])
    sessions = user_data["sessions"]

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    tmp_file = os.path.join(tmp_dir, f"{user_data['uuid']}.json")
    checkpoint_file = os.path.join(tmp_dir, f"{user_data['uuid']}_checkpoint.json")

    # Checkpoint resume: if a checkpoint exists, load it and skip completed sessions.
    completed_session_ids = set()
    new_user_data = {
        "uuid": user_data["uuid"],
        "user_name": user_name,
        "sessions": []
    }
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            new_user_data = json.load(f)
        completed_session_ids = {s["_checkpoint_idx"] for s in new_user_data["sessions"] if s.get("_checkpoint_idx") is not None}
        print(f"⚡ Resuming {user_name}: {len(completed_session_ids)} sessions already done")
    else:
        client.delete_all(user_id=user_name)

    try:

        for sess_idx, session in enumerate(tqdm(sessions, total=len(sessions), desc=f"Processing user {user_name}")):
            if sess_idx in completed_session_ids:
                continue
            new_session = {
                "memory_points": session["memory_points"],
                "dialogue": session["dialogue"]
            }

            # add messages
            dialogue = session["dialogue"]
            formatted_dialogue = [
                {
                    "role": turn["role"],
                    "content": turn["content"],
                }
                for turn in dialogue
            ]

            date_format = "%b %d, %Y, %H:%M:%S"
            dt = datetime.strptime(session["start_time"], date_format).replace(tzinfo=timezone.utc)
            iso_date = dt.isoformat()
            timestamp = int(dt.timestamp())

            result, duration_ms = add_memory(
                client=client, 
                user_id=user_name, 
                message=formatted_dialogue, 
                timestamp=timestamp
            )

            if session.get('is_generated_qa_session', False):
                new_session["add_dialogue_duration_ms"] = duration_ms
                new_session["is_generated_qa_session"] = True
                del new_session["dialogue"]
                del new_session["memory_points"]
                new_session["_checkpoint_idx"] = sess_idx
                new_user_data["sessions"].append(new_session)
                continue

            extracted_memories = [
                item["memory"] for item in result["results"]
            ]
            new_session["extracted_memories"] = extracted_memories
            new_session["add_dialogue_duration_ms"] = duration_ms

            # search updated memories
            for memory in new_session["memory_points"]:
                if memory["is_update"] == "False" or not memory["original_memories"]:
                    continue

                _, memories_from_system, duration_ms = search_memory(
                    client=client, 
                    query=memory["memory_content"],
                    user_id=user_name, 
                    top_k=10
                )

                memory["memories_from_system"] = memories_from_system

            # search and query
            if "questions" not in session:
                new_session["_checkpoint_idx"] = sess_idx
                new_user_data["sessions"].append(new_session)
                continue

            new_session["questions"] = []

            for qa in session["questions"]:

                context, _, duration_ms = search_memory(
                    client=client, 
                    query=qa["question"], 
                    user_id=user_name, 
                    top_k=top_k
                )

                new_qa = copy.deepcopy(qa)
                new_qa["context"] = context
                new_qa["search_duration_ms"] = duration_ms

                prompt = PROMPT_MEMZERO.format(
                    context=context,
                    question=qa["question"]
                )

                start_time = time.time()
                response = llm_request(prompt)
                new_qa["system_response"] = response
                new_qa["response_duration_ms"] = (time.time() - start_time) * 1000

                new_session["questions"].append(new_qa)
            
            new_session["_checkpoint_idx"] = sess_idx
            new_user_data["sessions"].append(new_session)

            # Save checkpoint after every session so a failure mid-run can resume.
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(new_user_data, f, ensure_ascii=False)

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_user_data, f, ensure_ascii=False)

        print(f"✅ Saved user {user_name} to {tmp_file}")
        return {"uuid": user_data["uuid"], "status": "ok", "path": tmp_file}

    except Exception as e:
        error_path = os.path.join(tmp_dir, f"{user_data['uuid']}_error.log")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"❌ Error in user {user_name}: {e}")
        return {"uuid": user_data["uuid"], "status": "error", "path": error_path}


def iter_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(
    data_path: str,
    version: str = "default",
    top_k: int = 20,
    max_workers: int = 2
):
    frame = "memzero"
    save_path = f"results/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(save_path, f"{frame}_eval_results.jsonl")
    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    start_time = time.time()

    for idx, user_data in enumerate(iter_jsonl(data_path), 1):
        total_users = idx
        try:
            result = process_user(user_data, top_k, save_path)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue