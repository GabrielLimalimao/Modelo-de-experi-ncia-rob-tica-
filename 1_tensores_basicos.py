"""Exemplo introdutório: o que é um tensor e quantas dimensões ele tem."""

import torch


def main() -> None:
    tensores = {
        "Scalar": torch.tensor(13.0),
        "Vetor": torch.tensor([1, 2, 3]),
        "Matriz": torch.tensor([[1, 2], [3, 4]]),
        "Cubo": torch.tensor([[[1, 2], [2, 3]], [[4, 5], [6, 7]]]),
    }

    for nome, tensor in tensores.items():
        print(
            f"{nome:<7} | formato={tuple(tensor.shape)} "
            f"| dimensões={tensor.ndim} | tipo={tensor.dtype} | {tensor.tolist()}"
        )


if __name__ == "__main__":
    main()
