def test_imports():
    import src.api.app  # noqa: F401
    import src.data.preprocess  # noqa: F401
    import src.data.split  # noqa: F401
    import src.models.inference  # noqa: F401
    import src.models.training  # noqa: F401
    import src.pipelines.infer_pipeline  # noqa: F401
    import src.pipelines.train_pipeline  # noqa: F401

