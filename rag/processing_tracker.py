import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
from config import Config

config = Config()

class ProcessingTracker:
    """追蹤檔案處理狀態，避免重複處理"""
    
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or os.path.join(config.EMBED_DIR, "processing_log.json")
        self.log_data = self._load_log()
    
    def _load_log(self) -> Dict:
        """載入處理記錄"""
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        # 檔案是空的，返回預設結構
                        return {
                            "files": {},
                            "last_full_rebuild": None,
                            "version": "1.0"
                        }
                    return json.loads(content)
            except (json.JSONDecodeError, Exception) as e:
                # JSON 解析失敗，備份舊檔案並建立新的
                print(f"⚠️  Warning: Could not load {self.log_path}: {str(e)}")
                backup_path = self.log_path + ".backup"
                if os.path.exists(self.log_path):
                    os.rename(self.log_path, backup_path)
                    print(f"   Old file backed up to {backup_path}")
                return {
                    "files": {},
                    "last_full_rebuild": None,
                    "version": "1.0"
                }
        return {
            "files": {},
            "last_full_rebuild": None,
            "version": "1.0"
        }
    
    def _save_log(self):
        """儲存處理記錄"""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """計算檔案的 SHA256 雜湊值"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_file_mtime(self, file_path: str) -> float:
        """取得檔案修改時間"""
        return os.path.getmtime(file_path)
    
    def is_file_processed(self, file_path: str, stage: str = "all") -> bool:
        """
        檢查檔案是否已處理過
        
        Args:
            file_path: 檔案路徑
            stage: 處理階段 ('chunking', 'embedding', 'indexing', 'all')
        
        Returns:
            True 如果檔案已處理且未變更
        """
        if file_path not in self.log_data["files"]:
            return False
        
        file_info = self.log_data["files"][file_path]
        
        # 檢查檔案是否存在
        if not os.path.exists(file_path):
            return True  # 檔案已刪除，視為已處理（跳過）
        
        # 計算當前檔案雜湊值
        current_hash = self._calculate_file_hash(file_path)
        
        # 如果雜湊值不同，表示檔案已變更
        if file_info.get("file_hash") != current_hash:
            return False
        
        # 檢查特定階段的處理狀態
        if stage == "all":
            return (file_info.get("chunking_status") == "completed" and
                    file_info.get("embedding_status") == "completed" and
                    file_info.get("index_status") == "completed")
        else:
            return file_info.get(f"{stage}_status") == "completed"
    
    def mark_file_processing(self, file_path: str, stage: str):
        """標記檔案開始處理某階段"""
        if file_path not in self.log_data["files"]:
            self.log_data["files"][file_path] = {
                "file_hash": self._calculate_file_hash(file_path),
                "file_mtime": self._get_file_mtime(file_path),
                "first_processed": datetime.now().isoformat()
            }
        
        self.log_data["files"][file_path][f"{stage}_status"] = "processing"
        self.log_data["files"][file_path][f"{stage}_started"] = datetime.now().isoformat()
        self._save_log()
    
    def mark_file_completed(self, file_path: str, stage: str, **kwargs):
        """
        標記檔案完成某階段處理
        
        Args:
            file_path: 檔案路徑
            stage: 處理階段
            **kwargs: 額外資訊（如 chunk_count, embedding_range 等）
        """
        if file_path not in self.log_data["files"]:
            self.log_data["files"][file_path] = {
                "file_hash": self._calculate_file_hash(file_path),
                "file_mtime": self._get_file_mtime(file_path),
                "first_processed": datetime.now().isoformat()
            }
        
        file_info = self.log_data["files"][file_path]
        file_info[f"{stage}_status"] = "completed"
        file_info[f"{stage}_completed"] = datetime.now().isoformat()
        file_info["last_processed"] = datetime.now().isoformat()
        
        # 儲存額外資訊
        for key, value in kwargs.items():
            file_info[key] = value
        
        self._save_log()
    
    def mark_file_failed(self, file_path: str, stage: str, error: str):
        """標記檔案處理失敗"""
        if file_path not in self.log_data["files"]:
            self.log_data["files"][file_path] = {
                "file_hash": self._calculate_file_hash(file_path) if os.path.exists(file_path) else None,
                "first_processed": datetime.now().isoformat()
            }
        
        self.log_data["files"][file_path][f"{stage}_status"] = "failed"
        self.log_data["files"][file_path][f"{stage}_error"] = error
        self.log_data["files"][file_path][f"{stage}_failed"] = datetime.now().isoformat()
        self._save_log()
    
    def get_unprocessed_files(self, directory: str, extension: str = ".txt", stage: str = "all") -> List[str]:
        """
        取得未處理的檔案列表
        
        Args:
            directory: 目錄路徑
            extension: 檔案副檔名
            stage: 處理階段
        
        Returns:
            未處理的檔案路徑列表
        """
        unprocessed = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(extension):
                    file_path = os.path.join(root, file)
                    if not self.is_file_processed(file_path, stage):
                        unprocessed.append(file_path)
        
        return unprocessed
    
    def get_file_info(self, file_path: str) -> Optional[Dict]:
        """取得檔案的處理資訊"""
        return self.log_data["files"].get(file_path)
    
    def remove_file_record(self, file_path: str):
        """移除檔案的處理記錄（當檔案被刪除時）"""
        if file_path in self.log_data["files"]:
            del self.log_data["files"][file_path]
            self._save_log()
    
    def mark_full_rebuild(self):
        """標記完整重建時間"""
        self.log_data["last_full_rebuild"] = datetime.now().isoformat()
        self._save_log()
    
    def get_statistics(self) -> Dict:
        """取得處理統計資訊"""
        total = len(self.log_data["files"])
        completed = sum(1 for f in self.log_data["files"].values() 
                       if f.get("chunking_status") == "completed" and
                          f.get("embedding_status") == "completed" and
                          f.get("index_status") == "completed")
        failed = sum(1 for f in self.log_data["files"].values()
                    if "failed" in str(f.get("chunking_status", "")) or
                       "failed" in str(f.get("embedding_status", "")) or
                       "failed" in str(f.get("index_status", "")))
        
        return {
            "total_files": total,
            "completed_files": completed,
            "failed_files": failed,
            "processing_files": total - completed - failed,
            "last_full_rebuild": self.log_data.get("last_full_rebuild")
        }
