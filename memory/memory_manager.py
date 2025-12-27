# memory/memory_manager.py
import time
import uuid
import sys
import json
from typing import List, Optional, Dict
from pathlib import Path

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import settings
from agent.schemas import MemoryItem, MemoryType, UserProfileSchema
from utils.llm_client import get_embedding
from utils.json_utils import safe_json_load 
from utils.logger import get_logger
from memory.chat_summarizer import ChatSummarizer
from memory.user_profile_store import UserProfileStore
from retrieval.semantic_cache import clear_cache  

try:
    from retrieval.pinecone_client import index
except ImportError:
    index = None

logger = get_logger("MEMORY_MANAGER")

class MemoryManager:
    """
    Hybrid Memory System:
    1. Short-Term (RAM): Linear context.
    2. Long-Term (Pinecone): Episodic archival & Semantic Facts.
    3. Profile (Pinecone): User personality.
    """
    NAMESPACE = "memory_store"

    def __init__(self):
        self.summarizer = ChatSummarizer()
        self.profile_store = UserProfileStore()
        
        # Short-Term Linear Buffer: {user_id: [{"role": "user", "content": "..."}, ...]}
        self._active_context: Dict[str, List[Dict[str, str]]] = {}
     
    def get_profile(self, user_id: str) -> UserProfileSchema:
        return self.profile_store.get_profile(user_id)

    def get_immediate_context(self, user_id: str, window_size: int = 4) -> str:
        history = self._active_context.get(user_id, [])
        if not history:
            return ""
        
        recent_turns = history[-window_size:]
        formatted_context = []
        for msg in recent_turns:
            role = "User" if msg['role'] == "user" else "Agent"
            formatted_context.append(f"{role}: {msg['content']}")
            
        return "\n".join(formatted_context)

    def clear_chat_history(self, user_id: str) -> bool:
        if user_id in self._active_context:
            del self._active_context[user_id]

        if not index: return False
        try:
            # Strictly scoped deletion
            index.delete(
                filter={"user_id": {"$eq": user_id}, "type": {"$eq": "episodic"}},
                namespace=self.NAMESPACE
            )
            return True
        except Exception as e:
            logger.error(f"Failed to clear episodic history: {e}")
            return False

    def reset_memory(self, user_id: str) -> bool:
        success = True
        if user_id in self._active_context:
            del self._active_context[user_id]

        if index: 
            try:
                # Strictly scoped deletion
                index.delete(
                    filter={"user_id": {"$eq": user_id}},
                    namespace=self.NAMESPACE
                )
                logger.warning(f"Full memory reset performed for user: {user_id}")
            except Exception as e:
                logger.error(f"Failed to reset memory vectors: {e}")
                success = False
        # 2. Delete Profile from Persistence
        self.profile_store.delete_profile(user_id)
        
        try:
            clear_cache()
        except Exception as e:
            logger.error(f"Failed to clear semantic cache during reset: {e}")

        return success

    def add_memory(self, content: str, memory_type: MemoryType, metadata: dict = None) -> Optional[str]:
        """
        Stores memory in Pinecone.
        ENFORCES user isolation: Checks if user_id is missing for user-specific memory types.
        """
        if not index or not content: return None
        
        safe_meta = metadata or {}
        
        # --- Strict Isolation Check ---
        # If storing facts or episodes, user_id is MANDATORY to prevent global leakage.
        if memory_type in [MemoryType.FACT, MemoryType.EPISODIC]:
            if "user_id" not in safe_meta:
                logger.error(f"Attempted to store {memory_type} without 'user_id'. Aborting to prevent global pollution.")
                return None

        try:
            vector = get_embedding(content)
            if not vector: return None
            
            mem_id = str(uuid.uuid4())
            pinecone_meta = {
                "text": content,
                "type": memory_type.value,
                "timestamp": str(time.time()),
                "attributes_json": json.dumps(safe_meta) # Store original metadata as JSON string
            }
            
            # Lift user_id to top-level metadata for filtering
            if "user_id" in safe_meta:
                pinecone_meta["user_id"] = safe_meta["user_id"]

            index.upsert(
                vectors=[{"id": mem_id, "values": vector, "metadata": pinecone_meta}],
                namespace=self.NAMESPACE
            )
            return mem_id
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            return None

    def retrieve_relevant(self, query: str, user_id: str, limit: int = 5, memory_type: Optional[MemoryType] = None, score_threshold: float = 0.70) -> List[MemoryItem]:
        """
        Semantic Retrieval with STRICT user_id filtering.
        """
        if not index or not query: return []
        try:
            query_vector = get_embedding(query)
            if not query_vector: return []

            # Always filter by user_id
            filter_dict = {"user_id": user_id}
            if memory_type: filter_dict["type"] = memory_type.value

            results = index.query(
                vector=query_vector, top_k=limit, include_metadata=True,
                namespace=self.NAMESPACE, filter=filter_dict
            )

            memories = []
            for match in results.get('matches', []):
                if match.score < score_threshold: continue
                
                # Handle object vs dict metadata
                meta = getattr(match, "metadata", None) or match.get("metadata", {})
                
                # Retrieve nested attributes
                attr = safe_json_load(meta.get("attributes_json", "{}")) or {}

                memories.append(MemoryItem(
                    id=match.id,
                    content=meta.get("text", ""),
                    memory_type=MemoryType(meta.get("type", "fact")),
                    timestamp=meta.get("timestamp", ""),
                    metadata=attr
                ))
            return memories
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    def check_and_update_profile_pre_planning(self, user_id: str, user_query: str):
        if not user_query: return 

        try:
            current_profile = self.profile_store.get_profile(user_id)
            original_dump = current_profile.model_dump()

            updates = self.summarizer.analyze_interaction_delta(
                current_profile=current_profile, 
                last_user_msg=user_query, 
                last_agent_msg="[SYSTEM: PRE-RESPONSE CHECK]"
            )

            if updates:
                updated_data = original_dump.copy()
                updated_data.update(updates)
                new_profile = UserProfileSchema(**updated_data)
                
                changed = self.profile_store.sync_if_changed(current_profile, new_profile)
                if changed:
                    logger.info(f"Profile updated based on pre-planning check for {user_id}")

        except Exception as e:
            logger.warning(f"Pre-planning profile check failed: {e}")

    def process_realtime_interaction(self, user_id: str, user_msg: str, agent_msg: str):
        if not user_msg or not agent_msg: return

        # 1. Update Short-Term Context
        if user_id not in self._active_context:
            self._active_context[user_id] = []
        
        self._active_context[user_id].append({"role": "user", "content": user_msg})
        self._active_context[user_id].append({"role": "agent", "content": agent_msg})
        
        if len(self._active_context[user_id]) > 20:
             self._active_context[user_id] = self._active_context[user_id][-20:]

        # 2. Analyze Profile Delta
        try:
            current_profile = self.profile_store.get_profile(user_id)
            original_dump = current_profile.model_dump()

            updates = self.summarizer.analyze_interaction_delta(
                current_profile=current_profile, 
                last_user_msg=user_msg, 
                last_agent_msg=agent_msg
            )

            if updates:
                updated_data = original_dump.copy()
                updated_data.update(updates)
                new_profile = UserProfileSchema(**updated_data)
                self.profile_store.sync_if_changed(current_profile, new_profile)

            # 3. Episodic Archival
            if len(user_msg) + len(agent_msg) > 50:
                self.add_memory(
                    content=f"User: {user_msg}\nAgent: {agent_msg}",
                    memory_type=MemoryType.EPISODIC,
                    metadata={"user_id": user_id}
                )

        except Exception as e:
            logger.error(f"Real-time processing failed: {e}")

    def consolidate_session(self, user_id: str, conversation_history: List[str]):
        """End-of-session bulk summarization with STRICT 25-FACT LIMIT and USER SCOPING."""
        if not conversation_history: return
        
        try:
            # 1. Extract raw facts
            chat_data = self.summarizer.summarize(conversation_history)
            if not chat_data.key_facts: return

            # 2. Fetch existing facts (Strictly scoped to user_id)
            existing_memories = self.retrieve_relevant(
                query="general user facts", 
                user_id=user_id, 
                limit=50, 
                memory_type=MemoryType.FACT,
                score_threshold=0.0 # Fetch ALL candidates
            )
            existing_texts = [m.content for m in existing_memories]

            # 3. Smart Deduplication
            final_facts = self.summarizer.deduplicate_facts(existing_texts, chat_data.key_facts)

            # 4. Replace Strategy: Delete OLD facts (Scoped to User), Insert FINAL set
            if index:
                # CRITICAL: Scope deletion to this user only
                index.delete(
                    filter={"user_id": {"$eq": user_id}, "type": {"$eq": "fact"}},
                    namespace=self.NAMESPACE
                )
                
                for fact in final_facts:
                    # add_memory now enforces user_id presence
                    self.add_memory(fact, MemoryType.FACT, {"user_id": user_id})
                
                logger.info(f"Consolidated facts for {user_id}. Final count: {len(final_facts)}")

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")