def check_positive(**kwargs: float):
    """
    >>> check_positive(width=10.5, height=2.0)
    >>> check_positive(length=5, depth=0)
    ValueError: depth is not positive
    """
    for name, val in kwargs.items():
        if val <= 0:
            raise ValueError(f"{name} is not positive")
