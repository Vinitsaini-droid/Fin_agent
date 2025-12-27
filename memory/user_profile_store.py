# memory/user_profile_store.py
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from agent.schemas import UserProfileSchema 
from config.settings import settings
from utils.json_utils import safe_json_load
from utils.logger import get_logger

try:
    from retrieval.pinecone_client import index
except ImportError:
    index = None

logger = get_logger("USER_PROFILE")

class UserProfileStore:
    """
    Manages persistence of the UserProfileSchema in Pinecone.
    """
    PROFILE_NAMESPACE = "user_profiles"
    
    def __init__(self):
        if not index:
            logger.warning("UserProfileStore: Pinecone unavailable (Local Mode).")

    def _parse_fetch_response(self, response: Any, user_id: str) -> Optional[Dict]:
        """
        Helper to safely extract vector data regardless of Pinecone SDK version
        (Dict vs Object response).
        """
        if not response:
            return None

        # Handle Object-based response (Newer SDKs)
        if hasattr(response, "vectors"):
            vectors = response.vectors
        # Handle Dict-based response (Older SDKs / JSON)
        elif isinstance(response, dict) and "vectors" in response:
            vectors = response["vectors"]
        else:
            return None

        # Check if user_id exists in the vectors map
        # Note: 'vectors' can be a dict or a map-like object
        if user_id in vectors:
            return vectors[user_id]
        
        return None

    def check_user_status(self, user_id: str) -> str:
        """
        Determines if a user has an existing profile in the database.
        Returns 'old' if profile exists, 'new' otherwise.
        """
        if not index:
            logger.info(f"Local Mode: Treating user {user_id} as 'new'.")
            return "new"

        try:
            # We use fetch to check for existence of the ID in the specific namespace
            result = index.fetch(ids=[user_id], namespace=self.PROFILE_NAMESPACE)
            
            vector_data = self._parse_fetch_response(result, user_id)
            if vector_data:
                logger.info(f"User {user_id} identified as 'old'.")
                return "old"
            
            logger.info(f"User {user_id} identified as 'new'.")
            return "new"
            
        except Exception as e:
            logger.error(f"Error checking user status for {user_id}: {e}")
            # Fail safe to 'new' to prevent crashing flow
            return "new"

    def get_profile(self, user_id: str) -> UserProfileSchema:
        """Fetch profile or return default."""
        if not index:
            return UserProfileSchema(user_id=user_id)

        try:
            result = index.fetch(ids=[user_id], namespace=self.PROFILE_NAMESPACE)
            
            vector_data = self._parse_fetch_response(result, user_id)
            if vector_data:
                # Handle object vs dict metadata access
                meta = getattr(vector_data, "metadata", None) or vector_data.get("metadata", {})
                
                # 'profile_data' is stored as a JSON string inside metadata
                raw_json = meta.get('profile_data')
                data = safe_json_load(raw_json)
                
                if data:
                    return UserProfileSchema.model_validate(data)
            
            return UserProfileSchema(user_id=user_id)
        except Exception as e:
            logger.error(f"Fetch profile error: {e}")
            return UserProfileSchema(user_id=user_id)

    def update_profile(self, profile: UserProfileSchema):
        """Force write to DB."""
        if not index: return
        try:
            # CRITICAL FIX: Ensure placeholder is always a valid list of floats
            # If embedding dimension is missing or 0, default to 768 or 1536 just to be safe, 
            # though usually settings should have it.
            dim = getattr(settings, "EMBEDDING_DIMENSION", 768)
            placeholder = [0.0] * dim
            if placeholder:
                placeholder[0] = 1.0 # Ensure non-zero vector

            # Serialize strictly
            profile_json = profile.model_dump_json()

            index.upsert(
                vectors=[{
                    'id': profile.user_id,
                    'values': placeholder,
                    'metadata': {
                        'profile_data': profile_json, 
                        'type': 'user_profile'
                    }
                }],
                namespace=self.PROFILE_NAMESPACE
            )
        except Exception as e:
            logger.error(f"Update profile error: {e}")

    def sync_if_changed(self, old_profile: UserProfileSchema, current_profile: UserProfileSchema) -> bool:
        """
        Smart Sync: Only writes if data actually changed.
        """
        if old_profile.model_dump() != current_profile.model_dump():
            logger.info(f"Syncing profile changes for {current_profile.user_id}...")
            self.update_profile(current_profile)
            return True
        return False

    def delete_profile(self, user_id: str) -> None:
        """
        Hard Delete: Removes the user profile from the database.
        """
        if not index: return
        try:
            logger.warning(f"DELETING PROFILE for {user_id}...")
            index.delete(ids=[user_id], namespace=self.PROFILE_NAMESPACE)
            logger.info(f"Profile deleted successfully for {user_id}")
        except Exception as e:
            logger.error(f"Failed to delete profile for {user_id}: {e}")