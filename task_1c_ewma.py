from _task_wrapper_bootstrap import main_for_task


if __name__ == "__main__":
    main_for_task("task_1c", imputation_method="ewma", generate_plots=False, export_intermediate=True)
