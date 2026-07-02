"""Learning layer (Fase 9): state encoding and self-play value/policy models.

Kept dependency-light for now -- only the encoder lives here and needs nothing
beyond numpy. Model training (PyTorch) is added later and only if the de-risk
probe shows a learned leaf can beat the heuristic (see TODO Fase 9).
"""
