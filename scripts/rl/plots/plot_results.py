"""
SCRATCH CODE
"""

from scripts.rl.plots.plot_utils import filter_df, plot_sweep
from src.wandb_utils import convert_runs_to_df, get_runs_from_wandb_projects


CONFIGS = ["train.epochs",
           "train.total_steps",
           "method.num_rollouts",
           "optimizer.kwargs.lr",
           "train.batch_size",
           "method.ppo_epochs",
           "method.init_kl_coef",
           "method.target",
           "model.model_path",
           "train.seed"]
KEYS = ["reward/mean"]

def plot_initial_runs():
        runs = get_runs_from_wandb_projects("sita/rl-meg")
        df = convert_runs_to_df(runs, keys=KEYS, configs=CONFIGS)

        seed_df = filter_df(df, total_steps=300, seed=None).drop_duplicates()
        print(seed_df)
        if len(seed_df) > 0:
                import matplotlib.pyplot as plt
                import matplotlib.ticker as mtick
                plt.figure(figsize=(6, 4))
                plt.boxplot(seed_df['reward/mean'], labels=[''])
                plt.ylim((0, 1))
                plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(0.1))
                plt.grid(axis="y", alpha=0.3)
                plt.title('[]')
                plt.suptitle('')
                plt.ylabel('Mean reward')
                plt.savefig("scripts/rl/plot.png")
                plt.show()
        else:
                print("No results found")
                
        lr_df = filter_df(df, total_steps=300, lr=None, seed=None).drop_duplicates()
        plot_sweep(lr_df, x_axis="optimizer.kwargs.lr", suptitle="", default_value=1e-5)
        ikc_df = filter_df(df, total_steps=300, init_kl_coef=None, seed=None).drop_duplicates()
        plot_sweep(ikc_df, x_axis="method.init_kl_coef", suptitle="", default_value=0.05)
        target_df = filter_df(df, total_steps=300, target=None, seed=None).drop_duplicates()
        plot_sweep(target_df, x_axis="method.target", suptitle="", default_value=6)
        

def plot_initial_assistant_runs():
        runs = get_runs_from_wandb_projects("rl-sweep-assistant-meg")
        df = convert_runs_to_df( runs, keys=KEYS, configs=CONFIGS)

        df_300 = filter_df(df, model_path=None, total_steps=300, seed=None).drop_duplicates()
        df_1000 = filter_df(df, model_path=None, total_steps=1000, seed=None).drop_duplicates()
        plot_sweep(df_300,
                df_1000,
                x_axis="model.model_path",
                suptitle="",
                labels=["300 total steps", "1000 total steps"],
                colors=['k', 'b'], 
                suffix="300vs1000")
        

def plot_some_assistant_runs():
        runs = get_runs_from_wandb_projects('rl-assistant-meg')
        df = convert_runs_to_df(runs, keys=KEYS, configs=CONFIGS)
        df_control = filter_df(df, model_path=["725725_0.", "725725_1."], total_steps=300, seed=None).drop_duplicates()
        df_treatment = filter_df(df, model_path=["725725_10.", "725725_11."], total_steps=300, seed=None).drop_duplicates()
        plot_sweep(df_control,
                   df_treatment,
                x_axis="model.model_path",
                suptitle="Initial assistant runs",
                title="[total_steps=300, lr=1e-5, init_kl_coef=0.05]\nreward=sentiment",
                labels=["control", "treatment"],
                colors=['k', 'b'],
                linestyles=['', ''],
                adjust_subplots_top=0.8,
                suffix="x")
        

def plot_more_assistant_runs():    
        runs = get_runs_from_wandb_projects('rl-sweep-assistant-meg')
        df = convert_runs_to_df(runs, keys=KEYS, configs=CONFIGS)
        df_control_300 = filter_df(df, model_path=[f"725725_{i}." for i in range(6)], total_steps=300, seed=None).drop_duplicates()
        df_control_1000 = filter_df(df, model_path=[f"725725_{i}." for i in range(6)], total_steps=1000, seed=None).drop_duplicates()
        df_treatment_300 = filter_df(df, model_path=[f"725725_{i}." for i in range(6, 12)], total_steps=300, seed=None).drop_duplicates()
        df_treatment_1000 = filter_df(df, model_path=[f"725725_{i}." for i in range(6, 12)], total_steps=1000, seed=None).drop_duplicates()
        plot_sweep(df_control_300,
                df_control_1000,
                df_treatment_300,
                df_treatment_1000,
                x_axis="model.model_path",
                suptitle="Assistant runs",
                title="[lr=1e-5, init_kl_coef=0.05]\nreward=sentiment",
                labels=["control [300 steps]", "control [1000 steps]", "treatment [300 steps]", "treatment [1000 steps]"],
                colors=['k', 'k', 'b', 'b'],
                linestyles=['-', 'dotted', '-', 'dotted'],
                adjust_subplots_top=0.8,
                use_short_naming=True,
                suffix="x")
        

def plot_lr_assistant_runs():
        runs = get_runs_from_wandb_projects('rl-sweep-assistant-meg')
        df = convert_runs_to_df(runs, keys=KEYS, configs=CONFIGS)
        df_control = filter_df(df, model_path=[f"725725_{i}." for i in range(6)], total_steps=300, lr=None, seed=None).drop_duplicates()
        df_treatment = filter_df(df, model_path=[f"725725_{i}." for i in range(6, 12)], total_steps=300, lr=None, seed=None).drop_duplicates()
        plot_sweep(df_control,
                df_treatment,
                x_axis="optimizer.kwargs.lr",
                suptitle="Assistant runs (lr sweep)",
                title="[total_steps=300, init_kl_coef=0.05]\nreward=sentiment",
                labels=["control", "treatment"],
                colors=['k', 'b'],
                linestyles=['-', '-'],
                default_value=1e-5,
                adjust_subplots_top=0.8,
                suffix="")
        

def plot_init_kl_coef_assistant_runs():
        runs = get_runs_from_wandb_projects('rl-sweep-assistant-meg')
        df = convert_runs_to_df(runs, keys=KEYS, configs=CONFIGS)
        df_control = filter_df(df, model_path=[f"725725_{i}." for i in range(6)], total_steps=300, init_kl_coef=None, seed=None).drop_duplicates()
        df_treatment = filter_df(df, model_path=[f"725725_{i}." for i in range(6, 12)], total_steps=300, init_kl_coef=None, seed=None).drop_duplicates()
        plot_sweep(df_control,
                df_treatment,
                x_axis="method.init_kl_coef",
                suptitle="Assistant runs (init_kl_coef sweep)",
                title="[total_steps=300, lr=1e-5]\nreward=sentiment",
                labels=["control", "treatment"],
                colors=['k', 'b'],
                linestyles=['-', '-'],
                default_value=0.05,
                adjust_subplots_top=0.8,
                suffix="")
        

if __name__ == "__main__":
        plot_lr_assistant_runs()
        plot_init_kl_coef_assistant_runs()