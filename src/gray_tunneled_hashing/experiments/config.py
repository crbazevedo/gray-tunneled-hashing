"""Configuration dataclasses for experimental setups."""

from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class LSHExperimentConfig:
    """Configuração para experimentos LSH vs. Random Projection."""
    
    # Dataset
    n_samples: int = 100
    n_queries: int = 20
    dim: int = 16
    k: int = 5  # Número de vizinhos
    
    # Encoding
    n_bits: int = 8
    n_codes: int = 32
    
    # LSH parameters
    lsh_family: str = "hyperplane"  # "hyperplane" | "p_stable"
    p_stable_w: float = 1.0  # Width parameter para p-stable
    
    # GTH parameters
    use_gth: bool = True
    block_size: int = 4
    max_two_swap_iters: int = 20
    num_tunneling_steps: int = 0
    mode: str = "two_swap_only"  # "trivial" | "two_swap_only" | "full"
    
    # Query parameters
    hamming_radius: int = 1  # 0, 1, 2, ...
    
    # Reproducibility
    random_state: int = 42
    n_runs: int = 3  # Número de runs para média
    
    # Validation
    validate_collisions: bool = True
    validate_recall: bool = True
    
    def validate(self) -> list[str]:
        """
        Valida ranges de parâmetros e retorna lista de erros.
        
        Returns:
            Lista de mensagens de erro (vazia se válido)
        """
        errors = []
        
        # Dataset constraints
        if self.n_samples <= 0:
            errors.append("n_samples must be > 0")
        if self.n_queries <= 0:
            errors.append("n_queries must be > 0")
        if self.dim <= 0:
            errors.append("dim must be > 0")
        if self.k <= 0:
            errors.append("k must be > 0")
        if self.k > self.n_samples:
            errors.append(f"k ({self.k}) must be <= n_samples ({self.n_samples})")
        
        # Encoding constraints
        if self.n_bits <= 0:
            errors.append("n_bits must be > 0")
        if self.n_codes <= 0:
            errors.append("n_codes must be > 0")
        if self.n_codes > 2 ** self.n_bits:
            errors.append(
                f"n_codes ({self.n_codes}) must be <= 2**n_bits ({2**self.n_bits})"
            )
        
        # LSH constraints
        if self.lsh_family not in ["hyperplane", "p_stable"]:
            errors.append(f"lsh_family must be 'hyperplane' or 'p_stable', got '{self.lsh_family}'")
        if self.p_stable_w <= 0:
            errors.append("p_stable_w must be > 0")
        
        # GTH constraints
        if self.block_size <= 0:
            errors.append("block_size must be > 0")
        if self.max_two_swap_iters < 0:
            errors.append("max_two_swap_iters must be >= 0")
        if self.num_tunneling_steps < 0:
            errors.append("num_tunneling_steps must be >= 0")
        if self.mode not in ["trivial", "two_swap_only", "full"]:
            errors.append(f"mode must be 'trivial', 'two_swap_only', or 'full', got '{self.mode}'")
        
        # Query constraints
        if self.hamming_radius < 0:
            errors.append("hamming_radius must be >= 0")
        if self.hamming_radius > self.n_bits:
            errors.append(f"hamming_radius ({self.hamming_radius}) should be <= n_bits ({self.n_bits})")
        
        # Reproducibility
        if self.n_runs <= 0:
            errors.append("n_runs must be > 0")
        
        return errors
    
    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serializa para JSON."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, d: dict) -> "LSHExperimentConfig":
        """Cria instância a partir de dicionário."""
        return cls(**d)
    
    @classmethod
    def from_json(cls, json_str: str) -> "LSHExperimentConfig":
        """Deserializa de JSON."""
        d = json.loads(json_str)
        return cls.from_dict(d)

