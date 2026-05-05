"""
One-time setup: creates all MongoDB collections and indexes for the RoboOps pipeline.
Run this once after Atlas cluster is ready.

Usage:
    source .venv/bin/activate
    export MONGO_URI="mongodb+srv://robops:<password>@cluster0.cuzjsc9.mongodb.net/robops?retryWrites=true&w=majority"
    python pipeline/phase_a/setup_mongo_collections.py
"""

import os
import sys

import pymongo

MONGO_URI = os.environ.get("MONGO_URI", "")

COLLECTIONS = {
    "telemetry": [
        [("timestamp", pymongo.ASCENDING)],
        [("architecture", pymongo.ASCENDING)],
        [("mean_confidence", pymongo.ASCENDING)],
        [("source", pymongo.ASCENDING)],
        [("robot_id", pymongo.ASCENDING)],
    ],
    "production_metrics": [
        [("timestamp", pymongo.ASCENDING)],
        [("architecture", pymongo.ASCENDING)],
        [("source", pymongo.ASCENDING)],
    ],
    "retrain_queue": [
        [("architecture", pymongo.ASCENDING)],
        [("retrain_priority", pymongo.ASCENDING)],
        [("processed", pymongo.ASCENDING)],
        [("timestamp", pymongo.ASCENDING)],
    ],
    "drift_events": [
        [("timestamp", pymongo.ASCENDING)],
        [("architecture", pymongo.ASCENDING)],
        [("psi_score", pymongo.DESCENDING)],
    ],
}


def main():
    if not MONGO_URI:
        print("ERROR: MONGO_URI environment variable not set.")
        sys.exit(1)

    print("Connecting to MongoDB Atlas...")
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
        client.admin.command("ping")
        print("Connected!\n")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    db = client["robops"]

    for coll_name, indexes in COLLECTIONS.items():
        coll = db[coll_name]
        coll.insert_one({"_init": True})
        coll.delete_one({"_init": True})
        for index_keys in indexes:
            coll.create_index(index_keys)
        print(f"  [{coll_name}] created with {len(indexes)} indexes")

    client.close()
    print("\nDone. All collections and indexes are ready.")
    print("Collections: telemetry, production_metrics, retrain_queue, drift_events")


if __name__ == "__main__":
    main()
