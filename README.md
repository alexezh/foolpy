FoolPy is an experimental framework exploring how computation can be decomposed into many MicroLMs—tiny language models that are built dynamically by the system as needed. The concept draws inspiration from Lisp, where much of the language is defined in terms of itself.

The project began as an attempt to train a small transformer model to perform basic arithmetic within a minimal vocabulary. While the initial approach didn’t succeed as expected, it led to a new paradigm that blends ideas from machine learning and Lisp-style compositional computation.

In its current form, FoolPy focuses on constructing MicroLMs that can compute the next operation in a sequence of arithmetic computations. The long-term goal is to achieve fully reliable composition, where combinations of two or more MicroLMs produce deterministic, correct results—paving the way for self-organizing, interpretable computational systems.

Ultimately, the vision for FoolPy is to create a system capable of second-order self-modelling—a model that can represent itself from another perspective (analogous to how a predator models itself from the prey’s point of view). I consider such a model a precursor to AGI, and the target is to run it efficiently—within roughly 20 watts of power.
