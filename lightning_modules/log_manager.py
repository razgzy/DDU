import os
import shutil

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback


class LogManager(Callback):
    def on_train_start(self, trainer, pl_module):
        ckptcallback = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)][0]
        ckpt_dir = ckptcallback.dirpath
        print("*"*30)
        print('checkpoint_dir', ckpt_dir)
        print("*"*30)
        if os.path.exists(ckpt_dir):
            ckpt_list = [filename for filename in os.listdir(ckpt_dir) if '.ckpt' in filename]
            print(f'Log and the following checkpoint exists:\n Log dir: {ckpt_dir}\n' + '\n'.join(
                f'[{i}] {filename}' for i, filename in enumerate(ckpt_list)))

            delete = ['delete', 'd']
            resume = ['resume', 'r']
            quit = ['quit', 'q']
            ans = ''
            n_files = len(ckpt_list)
            while not (ans in delete or ans in resume or ans in quit):
                ans = input(f'[Number of ckpt files: {n_files}]\n'	
                            f'Delete the existing log and start a new experiment? Resume? Quit? (d/r/q) << ').lower()
                if ans in delete:
                    shutil.rmtree(trainer.log_dir)
                    os.makedirs(trainer.log_dir)
                elif ans in resume:
                    if n_files == 0:
                        print('Any checkpoint files do not exist!')
                        ans = ''
                    else:
                        s = ''
                        if n_files > 1:
                            while not (s.isdigit() and int(s) in range(n_files)):
                                s = input(f'Select which checkpoint to load. [0-{n_files - 1}]<< ').lower()
                            ckpt_path = os.path.join(ckpt_dir, ckpt_list[int(s)])
                        else:
                            ckpt_path = os.path.join(ckpt_dir, ckpt_list[0])
                        trainer.resume_from_checkpoint = ckpt_path
                elif ans in quit:
                    raise ValueError('Stopped as the log exist for this experiment.')
        else:
            print(f'Starting a new experiment and logging at \n {os.path.expanduser(trainer.log_dir)}')

    def on_exception(self, trainer, pl_module, exception):
        ckptcallback = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)][0]
        ckpt_dir = ckptcallback.dirpath
        exp_dir = os.path.join(*ckpt_dir.split('/')[:-1])
        if exp_dir in trainer.log_dir:
            pass
        else:
            exp_dir = trainer.log_dir
        if pl_module.global_rank == 0:
            yes = ['yes', 'y']
            no = ['no', 'n']
            ans = ''
            while not (ans in no or ans in yes):
                ans = input('Save this run? (Y/N) << ').lower()
                if ans in no:
                    shutil.rmtree(exp_dir)
                    print(f'Deleted {exp_dir}')
                    # Delete ModelCheckpoint callbacks
                    trainer.callbacks = [c for c in trainer.callbacks if not isinstance(c, ModelCheckpoint)]
                elif ans in yes:
                    ckpt_path = os.path.join(ckpt_dir, 'interrupted_model.ckpt')
                    trainer.save_checkpoint(ckpt_path)
                    print('Saved a checkpoint...')
