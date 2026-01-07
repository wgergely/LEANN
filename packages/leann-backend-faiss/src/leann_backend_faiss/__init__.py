import logging
import pickle
from pathlib import Path
from typing import Any, Literal, Optional, Union

import faiss
import numpy as np
from leann.interface import (
    LeannBackendBuilderInterface,
    LeannBackendFactoryInterface,
    LeannBackendSearcherInterface,
)
from leann.registry import register_backend

logger = logging.getLogger(__name__)


class FaissBackendBuilder(LeannBackendBuilderInterface):
    """FAISS-based index builder with GPU acceleration."""

    def build(self, data: np.ndarray, ids: list[str], index_path: str, **kwargs) -> None:
        """Build FAISS index on GPU."""
        logger.info(f"Building FAISS index with shape {data.shape}")

        d = data.shape[1]

        # Use GPU resources
        try:
            res = faiss.StandardGpuResources()
            logger.info("FAISS: GPU resources initialized")
            use_gpu = True
        except Exception as e:
            logger.warning(f"FAISS: Could not initialize GPU resources: {e}. Falling back to CPU.")
            use_gpu = False

        # Create index
        # For small datasets (<10k), Flat is best. For larger, IVFFlat.
        # User requested CAGRA, but that requires specific builds.
        # We'll use a robust heuristic.
        metric = faiss.METRIC_INNER_PRODUCT  # Default to cosine/IP for embeddings

        if use_gpu:
            try:
                # Try to use a flat GPU index for highest accuracy on small-medium data
                # Or IVFFlat for larger data.
                # For simplicity and speed on <1M vectors, Flat (Brute Force) on GPU is incredibly fast.
                if data.shape[0] < 100000:
                    config = faiss.GpuIndexFlatConfig()
                    config.useFloat16 = True
                    index = faiss.GpuIndexFlatIP(res, d, config)
                    logger.info("FAISS: Created GpuIndexFlatIP")
                else:
                    # IVF for larger datasets
                    nlist = int(np.sqrt(data.shape[0]))
                    index = faiss.index_factory(d, f"IVF{nlist},Flat", metric)
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    logger.info(f"FAISS: Created GPU IVF{nlist},Flat index")
            except Exception as e:
                logger.error(f"FAISS: Failed to create GPU index: {e}")
                raise
        else:
            index = faiss.IndexFlatIP(d)

        # normalize if using cosine/IP
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        faiss.normalize_L2(data)

        # Train if needed (IVF)
        if not index.is_trained:
            index.train(data)

        # Add vectors
        index.add(data)
        logger.info(f"FAISS: Added {index.ntotal} vectors to index")

        # Save index
        # GPU indices must be converted to CPU to save
        if use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(index)
        else:
            index_cpu = index

        # Save FAISS index
        index_file = Path(index_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index_cpu, str(index_file))

        # Save IDs separately
        ids_file = index_file.with_suffix(".ids.pkl")
        with open(ids_file, "wb") as f:
            pickle.dump(ids, f)
        logger.info(f"FAISS: Saved index to {index_file} and IDs to {ids_file}")


class FaissBackendSearcher(LeannBackendSearcherInterface):
    """FAISS-based searcher with GPU acceleration."""

    def __init__(self, index_path: str, **kwargs):
        self.index_path = Path(index_path)
        logger.info(f"FAISS: Loading index from {self.index_path}")

        # Load metadata to get embedding config
        meta_path = f"{self.index_path}.meta.json"
        try:
            import json

            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            self.embedding_model = meta.get("embedding_model", "facebook/contriever")
            self.embedding_mode = meta.get("embedding_mode", "sentence-transformers")
        except Exception as e:
            logger.warning(f"FAISS: Could not load metadata from {meta_path}: {e}")
            self.embedding_model = "facebook/contriever"
            self.embedding_mode = "sentence-transformers"

        # Load index
        self.index_cpu = faiss.read_index(str(self.index_path))

        # Load IDs
        ids_file = self.index_path.with_suffix(".ids.pkl")
        with open(ids_file, "rb") as f:
            self.ids = pickle.load(f)

        # Move to GPU if available
        try:
            self.res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index_cpu)
            logger.info("FAISS: Moved index to GPU")
        except Exception as e:
            logger.warning(f"FAISS: Could not move index to GPU: {e}. Using CPU.")
            self.index = self.index_cpu

    def _ensure_server_running(
        self, passages_source_file: str, port: Optional[int], **kwargs
    ) -> int:
        # FAISS searcher doesn't manage external servers explicitly,
        # but we need to return the port if it's expected by compute_query_embedding
        # For now, return the passed port or default
        return port if port else 5557

    def compute_query_embedding(
        self,
        query: str,
        use_server_if_available: bool = True,
        zmq_port: int = None,
        query_template: str = None,
        **kwargs,
    ) -> np.ndarray:
        # Import here to avoid circular dependency
        from leann.api import compute_embeddings

        # Apply template if provided
        if query_template:
            query = f"{query_template}{query}"

        # Force in-process computation to avoid ZMQ deadlocks since we don't manage a server yet
        return compute_embeddings(
            [query],
            model_name=self.embedding_model,
            mode=self.embedding_mode,
            use_server=False,
            port=None,
        )

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        **kwargs,
    ) -> dict[str, Any]:
        """Search for nearest neighbors."""

        # Normalize query for cosine similarity
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        faiss.normalize_L2(query)

        # Search
        distances, indices = self.index.search(query, top_k)

        # Map indices to IDs
        # indices is (B, K)
        results_labels = []
        results_distances = []

        for i in range(query.shape[0]):
            row_labels = []
            row_dists = []
            for j in range(top_k):
                idx = indices[i][j]
                if idx != -1:
                    row_labels.append(self.ids[idx])
                    row_dists.append(float(distances[i][j]))
            results_labels.append(row_labels)
            results_distances.append(row_dists)

        return {"labels": results_labels, "distances": results_distances}


@register_backend("faiss")
class FaissBackendFactory(LeannBackendFactoryInterface):
    """Factory for FAISS backend."""

    @staticmethod
    def builder(**kwargs) -> LeannBackendBuilderInterface:
        return FaissBackendBuilder()

    @staticmethod
    def searcher(index_path: str, **kwargs) -> LeannBackendSearcherInterface:
        return FaissBackendSearcher(index_path, **kwargs)
