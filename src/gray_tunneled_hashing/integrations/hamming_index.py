"""Hamming distance index for binary codes (FAISS + fallback)."""

import numpy as np
from typing import Optional, Tuple


class HammingIndex:
    """
    Wrapper for Hamming distance index.
    
    Supports both FAISS (if available) and pure Python fallback.
    """
    
    def __init__(
        self,
        codes: np.ndarray,
        use_faiss: bool = True,
        backend: Optional[str] = None,
    ):
        """
        Initialize Hamming index.
        
        Args:
            codes: Binary codes of shape (N, n_bits) with dtype bool or uint8
            use_faiss: If True, try to use FAISS (fallback to Python if not available)
            backend: Explicit backend to use ('faiss' or 'python'). If None, auto-detect.
        """
        if codes.ndim != 2:
            raise ValueError(f"Expected 2D codes, got shape {codes.shape}")
        
        # Convert bool to uint8 if needed (FAISS requires uint8)
        if codes.dtype == bool:
            self.codes = codes.astype(np.uint8)
        elif codes.dtype == np.uint8:
            self.codes = codes
        else:
            raise ValueError(
                f"Codes must be bool or uint8, got dtype {codes.dtype}"
            )
        
        self.n_samples, self.n_bits = self.codes.shape
        self.backend = backend
        self._index = None
        
        # Try to initialize with requested backend
        if backend is None:
            if use_faiss:
                try:
                    self._init_faiss()
                    self.backend = "faiss"
                except ImportError:
                    self._init_python()
                    self.backend = "python"
            else:
                self._init_python()
                self.backend = "python"
        elif backend == "faiss":
            self._init_faiss()
        else:  # python
            self._init_python()
    
    def _init_faiss(self):
        """Initialize FAISS binary index."""
        try:
            import faiss
            
            # FAISS requires codes to be packed (n_bits must be multiple of 8)
            # or we can use IndexBinaryFlat which handles it
            if self.n_bits % 8 != 0:
                # Pad to multiple of 8
                pad_bits = 8 - (self.n_bits % 8)
                padded_codes = np.pad(
                    self.codes,
                    ((0, 0), (0, pad_bits)),
                    mode="constant",
                    constant_values=0,
                )
                self._n_bits_padded = padded_codes.shape[1]
            else:
                padded_codes = self.codes
                self._n_bits_padded = self.n_bits
            
            # Convert to bytes (each row is n_bits/8 bytes)
            bytes_per_code = self._n_bits_padded // 8
            # Reshape to (N * n_bits_padded,), then pack bits in groups of 8
            codes_flat = padded_codes.reshape(-1)
            codes_bytes = np.packbits(codes_flat.reshape(-1, 8), axis=1).reshape(
                self.n_samples, bytes_per_code
            )
            
            # Create FAISS index
            index = faiss.IndexBinaryFlat(self._n_bits_padded)
            index.add(codes_bytes)
            
            self._index = index
            self._codes_bytes = codes_bytes
            
        except ImportError:
            raise ImportError(
                "FAISS not available. Install with: pip install faiss-cpu"
            )
    
    def _init_python(self):
        """Initialize pure Python index (no external dependencies)."""
        # Store codes as-is (uint8)
        self._index = None  # No external index needed
    
    def search(
        self,
        query_codes: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_codes: Query binary codes of shape (Q, n_bits) with dtype bool or uint8
            k: Number of neighbors to retrieve
            
        Returns:
            Tuple of (indices, distances) where:
            - indices: Array of shape (Q, k) with indices of nearest neighbors
            - distances: Array of shape (Q, k) with Hamming distances
        """
        if query_codes.ndim != 2:
            raise ValueError(f"Expected 2D query codes, got shape {query_codes.shape}")
        
        if query_codes.shape[1] != self.n_bits:
            raise ValueError(
                f"Query codes must have {self.n_bits} bits, got {query_codes.shape[1]}"
            )
        
        if k > self.n_samples:
            raise ValueError(
                f"k={k} cannot exceed number of samples={self.n_samples}"
            )
        
        # Convert bool to uint8 if needed
        if query_codes.dtype == bool:
            query_codes = query_codes.astype(np.uint8)
        
        if self.backend == "faiss":
            return self._search_faiss(query_codes, k)
        else:
            return self._search_python(query_codes, k)
    
    def _search_faiss(
        self,
        query_codes: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS."""
        import faiss
        
        # Pad query codes if needed
        if self.n_bits != self._n_bits_padded:
            query_codes_padded = np.pad(
                query_codes,
                ((0, 0), (0, self._n_bits_padded - self.n_bits)),
                mode="constant",
                constant_values=0,
            )
        else:
            query_codes_padded = query_codes
        
        # Convert to bytes
        Q = query_codes_padded.shape[0]
        bytes_per_code = self._n_bits_padded // 8
        query_flat = query_codes_padded.reshape(-1)
        query_bytes = np.packbits(query_flat.reshape(-1, 8), axis=1).reshape(
            Q, bytes_per_code
        )
        
        # Search
        distances, indices = self._index.search(query_bytes, k)
        
        return indices, distances
    
    def _search_python(
        self,
        query_codes: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search using pure Python (Hamming distance via XOR)."""
        Q = query_codes.shape[0]
        
        # Compute Hamming distances: XOR + count non-zero bits
        # query_codes: (Q, n_bits), self.codes: (N, n_bits)
        # distances: (Q, N)
        distances = np.zeros((Q, self.n_samples), dtype=np.int32)
        
        for i in range(Q):
            # XOR to find differing bits, then count
            xor_result = np.bitwise_xor(query_codes[i], self.codes)
            # Count non-zero bits (Hamming distance)
            distances[i] = np.count_nonzero(xor_result, axis=1)
        
        # Get k nearest (smallest distances)
        indices = np.argsort(distances, axis=1)[:, :k]
        
        # Get corresponding distances
        distances_sorted = np.take_along_axis(
            distances, indices, axis=1
        )
        
        return indices, distances_sorted


def build_hamming_index(
    codes: np.ndarray,
    use_faiss: bool = True,
    backend: Optional[str] = None,
) -> HammingIndex:
    """
    Build a Hamming distance index for binary codes.
    
    Args:
        codes: Binary codes of shape (N, n_bits) with dtype bool or uint8
        use_faiss: If True, try to use FAISS (fallback to Python if not available)
        backend: Explicit backend ('faiss' or 'python'). If None, auto-detect.
        
    Returns:
        HammingIndex instance
    """
    return HammingIndex(codes, use_faiss=use_faiss, backend=backend)


def search_hamming_index(
    index: HammingIndex,
    query_codes: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search a Hamming index for k nearest neighbors.
    
    Args:
        index: HammingIndex instance
        query_codes: Query binary codes of shape (Q, n_bits)
        k: Number of neighbors to retrieve
        
    Returns:
        Tuple of (indices, distances) where:
        - indices: Array of shape (Q, k) with indices of nearest neighbors
        - distances: Array of shape (Q, k) with Hamming distances
    """
    return index.search(query_codes, k)

