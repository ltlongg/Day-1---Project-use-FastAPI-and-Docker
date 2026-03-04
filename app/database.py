import os

from pymongo import MongoClient

DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://host.docker.internal:27017")

client = MongoClient(DATABASE_URL)
db = client.textclassifier

def get_db():
    """Dependency that provides the MongoDB database instance."""
    return db
