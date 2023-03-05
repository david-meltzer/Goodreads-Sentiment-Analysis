def log_raw_data(cfg):
    """
    Downloads Goodreads dataset from Kaggle and logs it as an artifact.

    Parameters
    ----------
        cfg (ConfigDict): ConfigDict object containing configuration for experiment.
    
    Returns:
    --------
        None

    """
    with wandb.init(
        project=cfg.PROJECT_NAME,job_type=cfg.RAW_DATA_JOB_TYPE,
        config=dict(cfg)
    ) as run:
        cfg=wandb.config
        os.system('kaggle competitions download -c goodreads-books-reviews-290312')
        os.system('tar -xopf goodreads-books-reviews-290312.zip')