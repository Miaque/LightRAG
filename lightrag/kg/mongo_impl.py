import os
from dataclasses import dataclass
from typing import Dict, Union

from pymongo import MongoClient
from tqdm.asyncio import tqdm as tqdm_async

from lightrag.base import (
    BaseKVStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from lightrag.utils import logger


@dataclass
class MongoKVStorage(BaseKVStorage):
    def __post_init__(self):
        client = MongoClient(
            os.environ.get("MONGO_URI", "mongodb://root:root@localhost:27017/")
        )
        database = client.get_database(os.environ.get("MONGO_DATABASE", "LightRAG"))
        collection_name = self.global_config["case_code"] + "_" + self.namespace
        self._data = database.get_collection(collection_name)
        logger.info(f"Use MongoDB as KV {collection_name}")

    async def all_keys(self) -> list[str]:
        return [x["_id"] for x in self._data.find({}, {"_id": 1})]

    async def get_by_id(self, id):
        return self._data.find_one({"_id": id})

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return list(self._data.find({"_id": {"$in": ids}}))
        return list(
            self._data.find(
                {"_id": {"$in": ids}},
                {field: 1 for field in fields},
            )
        )

    async def filter_keys(self, data: list[str]) -> set[str]:
        existing_ids = [
            str(x["_id"]) for x in self._data.find({"_id": {"$in": data}}, {"_id": 1})
        ]
        return set([s for s in data if s not in existing_ids])

    async def upsert(self, data: dict[str, dict]):
        for k, v in tqdm_async(data.items(), desc="Upserting"):
            self._data.update_one({"_id": k}, {"$set": v}, upsert=True)
            data[k]["_id"] = k
        return data

    async def drop(self):
        """ """
        pass


@dataclass
class MongoDocStatusStorage(DocStatusStorage):
    def __post_init__(self):
        client = MongoClient(
            os.environ.get("MONGO_URI", "mongodb://root:root@localhost:27017/")
        )
        database = client.get_database(os.environ.get("MONGO_DATABASE", "LightRAG"))
        self._col = database.get_collection(self.namespace)
        self.case_code = self.global_config["case_code"]

    async def filter_keys(self, data: list[str]) -> set[str]:
        # 根据给定data，查询"_id": {"$in": data} 并且 "status": {"$ne": "processed"}, case_code
        query = {
            "_id": {"$in": data},
            "status": {"$ne": DocStatus.PROCESSED},
            "case_code": self.case_code,
        }
        exist_ids = set([x["_id"] for x in self._col.find(query, {"_id": 1})])
        # exist_ids 和 data 计算差集
        return set(data).difference(exist_ids)

    async def get_status_counts(self) -> Dict[str, int]:
        # 根据status分组，统计每个status的文档数量，并加上case_code条件
        counts = self._col.aggregate(
            [
                {"$match": {"case_code": self.case_code}},
                {"$group": {"_id": "$status", "count": {"$sum": 1}}},
            ]
        )
        return {result["_id"]: result["count"] for result in counts}

    async def get_failed_docs(self) -> Dict[str, DocProcessingStatus]:
        """Get all failed documents"""
        query = {"status": DocStatus.FAILED, "case_code": self.case_code}
        return {doc["_id"]: doc for doc in self._col.find(query)}

    async def get_pending_docs(self) -> Dict[str, DocProcessingStatus]:
        """Get all pending documents"""
        query = {"status": DocStatus.PENDING, "case_code": self.case_code}
        return {doc["_id"]: doc for doc in self._col.find(query)}

    async def index_done_callback(self):
        """Save data to file after indexing"""
        pass

    async def upsert(self, data: dict[str, dict]):
        """Update or insert document status

        Args:
            data: Dictionary of document IDs and their status data
        """
        for k, v in data.items():
            v["_id"] = k
            v["case_code"] = self.case_code
            self._col.update_one({"_id": k}, {"$set": v}, upsert=True)
        return data

    async def get(self, doc_id: str) -> Union[DocProcessingStatus, None]:
        """Get document status by ID"""
        return self._col.find_one({"_id": doc_id, "case_code": self.case_code})

    async def delete(self, doc_ids: list[str]):
        """Delete document status by IDs"""
        self._col.delete_many({"_id": {"$in": doc_ids}, "case_code": self.case_code})
