# DDPModel.py

import torch.nn as nn
import torch
from copy import deepcopy
import datetime
import time
from tqdm import tqdm
import numpy as np
import inspect

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import all_reduce, ReduceOp, gather

# fixme an easy fix for running this seamlessly on cpus is use self.device <= 0 instead of self.device == 0
# since -1 I think its the cpu


class Model:

    def __init__(self, model_module: nn.Module,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 ddp: bool = False, # if True, the model will be wrapped in DDP, using device as the local rank.
                 world_size: int = 1,
                 compile_graph = False,
                 ema_decay = None,  # if set (e.g. 0.999), maintains an exponential moving average of the weights, updated once per training step. None disables EMA (default, matches previous behaviour).
                 # NOTE: EMA updates are currently only wired into trainDiffusionModel's loop.
                 # If ema_decay is set and the model is trained via trainModel instead, ema_state_dict stays
                 # frozen at its random initialization forever - no error, no warning, just stale/useless weights.
                 ):

        self.optimizer = None
        self.scheduler = None
        self.ddp = ( ddp and (world_size > 1) )
        self.device = device
        self.world_size = world_size
        self.bestLoss = torch.inf
        self.losses = None
        self.lossesTest = None
        self.trained = False
        self.ema_decay = ema_decay

        # get all additional arguments of the forward method, except self and x:
        self.additional_forward_args = inspect.getfullargspec(model_module.forward)[0][2:]


        self.model = model_module.to(self.device)


        if compile_graph:
            self.model = torch.compile(self.model, backend="inductor", mode="max-autotune", fullgraph=False)


        if self.ddp:
            self.model = DDP(self.model, device_ids = [self.device])
            self.bestmodelStateDict = deepcopy( self.model.module.state_dict() )
        else:
            self.bestmodelStateDict = deepcopy( self.model.state_dict() )
        
        self.ema_state_dict = deepcopy( self._base_model().state_dict() ) if self.ema_decay is not None else None
        self.best_ema_state_dict = deepcopy( self.ema_state_dict ) if self.ema_state_dict is not None else None


    def _base_model(self):
        # returns the underlying nn.Module, unwrapping DDP if needed
        return self.model.module if self.ddp else self.model


    def _update_ema(self):
        if self.ema_decay is None:
            return
        with torch.no_grad():
            for k, v in self._base_model().state_dict().items():
                if torch.is_floating_point(v):
                    self.ema_state_dict[k].mul_(self.ema_decay).add_(v, alpha = 1 - self.ema_decay)
                else:
                    self.ema_state_dict[k].copy_(v)


    def load_ema_as_weights(self, best = True):
        """Copies EMA shadow weights into the live model, in place. best=True (default)."""
        if self.ema_state_dict is None:
            raise ValueError("EMA is not enabled for this model (ema_decay was not set in the constructor).")
        state_dict_to_load = self.best_ema_state_dict if best else self.ema_state_dict
        self._base_model().load_state_dict(state_dict_to_load)


    def save_ema_state_dict(self, path, best = True, verbose = True):
        """"""
        if self.ema_state_dict is None:
            raise ValueError("EMA is not enabled for this model (ema_decay was not set in the constructor).")
        state_dict_to_save = self.best_ema_state_dict if best else self.ema_state_dict
        if (not self.ddp) or self._is_main_device():
            torch.save(state_dict_to_save, path)
            if verbose:
                print("EMA model parameters saved to " + path)




    def compile_graph(self, backend="inductor", mode="max-autotune", fullgraph=False):
        """
        Compile the underlying model for faster execution.
        Call this AFTER loading weights (recommended).
        """
        if self.ddp:
            # compile the wrapped module, keep DDP wrapper
            self.model.module = torch.compile(
                self.model.module, backend=backend, mode=mode, fullgraph=fullgraph
            )
        else:
            self.model = torch.compile(
                self.model, backend=backend, mode=mode, fullgraph=fullgraph
            )
            
            
        
    def set_optimizer(self, optimizer, **kwargs):
        if self.optimizer is not None:
            print("Optimizer changed. Rerun set_scheduler(). It has been set to None.")
            self.scheduler = None
            
        self.optimizer = optimizer(params=self.model.parameters(), **kwargs)
        
        
        
    def set_scheduler(self, scheduler, **kwargs):
        if self.optimizer is not None:
            self.scheduler = scheduler(optimizer = self.optimizer, **kwargs)
        else:
            raise ValueError("Optimizer must be set first")
        
        
        
    def train_iter(self, x, y, loss_function, **forward_kwargs) -> torch.Tensor:

        training_mode = self.model.training
        if not training_mode:
            self.model = self.model.train()
        
        x = x.to(self.device)
        
        for key in forward_kwargs:
            forward_kwargs[key] = forward_kwargs[key].to(self.device)
        
        y = y.to(self.device)
        
        
        # Perform one step of training:        
        self.optimizer.zero_grad()
                    
        output = self.model(x, **forward_kwargs)
        
        loss = loss_function(y, output)
            
        loss.backward()
        
        self.optimizer.step()
        
        
        self.model = self.model.train(mode = training_mode)
        
        return loss.item()
    
    
    
    def test_iter(self, x, y, loss_function, 
                  verbose = True, **forward_kwargs) -> torch.Tensor:
        training_mode = self.model.training
        if training_mode:
           self.model = self.model.eval() 
        
        x = x.to(self.device)
        for key in forward_kwargs:
            forward_kwargs[key] = forward_kwargs[key].to(self.device)
        
        
        y = y.to(self.device)

        
        with torch.no_grad():
            output = self.model(x, **forward_kwargs)
            loss = loss_function(y, output)
                                
        if verbose:
            print(f"Loss Test (acc): {loss}")                                
            
            
        self.model = self.model.train(mode = training_mode)
        
    
        return loss.item()
    
    
    
    def predict(self, x, **forward_kwargs):
        
        original_device = x.device
        
        training_mode = self.model.training
        if training_mode:
           self.model = self.model.eval() 
        
        x = x.to(self.device)
                
        for key in forward_kwargs:
            forward_kwargs[key] = forward_kwargs[key].to(self.device)
        
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x, **forward_kwargs)
            
            
        self.model = self.model.train(mode = training_mode)   
    
        
        return output.to(original_device)
    
    
    
    def updateBestModelandLoss(self, loss):
        
        if loss < self.bestLoss:
            if self.ddp:
                # If using DDP, we need to get the state dict from the module
                self.bestmodelStateDict = deepcopy(self.model.module.state_dict())
            else:
                # If not using DDP, we can directly access the state dict
                # Note: deepcopy is used to ensure we don't modify the original state dict
                self.bestmodelStateDict = deepcopy(self.model.state_dict())
            
            if self.ema_state_dict is not None:
                # freeze EMA averaging at this point (best val epoch); otherwise, it 
                # keeps on drifting
                self.best_ema_state_dict = deepcopy(self.ema_state_dict)

            self.bestLoss = loss
        
    

    
    def scheduler_step(self, loss, verbose = True):
        if (self.scheduler is not None):
            last_lr = self.scheduler.get_last_lr()
            self.scheduler.step(loss)
            if verbose and self._is_main_device():
                if last_lr != self.scheduler.get_last_lr():
                    print(f"Scheduler changed LR from {last_lr} to {self.scheduler.get_last_lr()}")
        
        else:
            if verbose:
                print("Scheduler not set. Ignoring scheduler step.")
                
                
                

    def load_state_dict(self, state_dict, last_loss=None, weights_only=True):
        if self.ddp:
            self.model.module.load_state_dict(
                torch.load(state_dict, weights_only=weights_only, map_location=self.device)
            )
        else:
            self.model.load_state_dict(
                torch.load(state_dict, weights_only=weights_only, map_location=self.device)
            )
        if last_loss is not None:
            self.bestLoss = last_loss
        self.trained = True
        
    
    
    def save_state_dict(self, path, best = True, verbose = True):
        # Only save on rank 0 in DDP
        if self.ddp:
            if self._is_main_device():
                if best:
                    torch.save(self.bestmodelStateDict, path)
                    if verbose:
                        print("Model parameters saved to " + path )
                else:
                    torch.save(self.model.module.state_dict(), path)
                    if verbose:
                        print("Model parameters saved to " + path )
            else:
                pass
        else:   
            if best:
                torch.save(self.bestmodelStateDict, path)
                if verbose:
                    print("Model parameters saved to " + path )
            else:
                torch.save(self.model.state_dict(), path)
                if verbose:
                    print("Model parameters saved to " + path )



    def plotComputationGraph(self, file_name = None, format = "png"):
        # fixme: this currently requires to properly get the shape
        from torchviz import make_dot

        x = torch.randn(1, *self.model.input_shape).to(self.device)  # Adjust input shape as needed
        y = self.model(x)

        dot = make_dot(y, params=dict(self.model.named_parameters()))
        dot.format = format

        if file_name is not None:
            dot.render(file_name)  # writes pytorch_graph.png
            print("Computation graph saved to " + file_name + "." + format)
            
        return dot


    def trainModel(self, 
                   my_loss_function, train_dataloader,
                   earlyStop_dataloader = None,
                   max_epochs = 1000, patience = 50, 
                   saveModelEvery = torch.inf,
                   write_losses = False,                   
                   folder_temp = None, 
                   final_model_name = None,
                   verbose = True,
                   save_best_model = True):
        
        """
        Trains the model using the provided training and early stopping dataloaders, with support for early stopping, 
        learning rate scheduling, and periodic model checkpointing.
        Args:
            my_loss (callable): The loss function to use for training and validation. Format: loss_function(y, output)
            train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset. Must contain at least two tensors in each batch:
                - [0]: input
                - [-1]: target tensor (e.g. labels or ground truth)
                - Additional tensors can be passed to the model's forward pass, will use **forward_kwargs.
            earlyStop_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset used for early stopping. Format is the same as train_dataloader.
            If set to None, no early stopping is performed.
            max_epochs (int, optional): Maximum number of epochs to train. Defaults to 1000.
            patience (int, optional): Number of epochs to wait for improvement in validation loss before stopping early. Defaults to 50.
            saveModelEvery (int or float, optional): Frequency (in epochs) to save model checkpoints. If set to torch.inf, disables periodic saving. Defaults to torch.inf.
            write_losses (bool, optional): Whether to save the training and validation loss arrays to disk after each epoch. Defaults to False.
            folder_temp (str, optional): Directory to save model checkpoints and loss arrays. Required if saveModelEvery is not torch.inf.
            final_model_name (str, optional): Path to save the final best model after training. If None, the model is not saved at the end. Defaults to None.
            verbose (bool, optional): Whether to print progress and status messages during training. Defaults to True.
        Raises:
            ValueError: If saveModelEvery is not torch.inf and folder_temp is not provided.
            ValueError: If both max_epochs and patience are set to torch.inf.
        Side Effects:
            - Updates self.losses and self.lossesTest with training and validation losses.
            - Updates self.bestLoss and self.bestmodelStateDict with the best validation loss and corresponding model parameters.
            - Saves model checkpoints and loss arrays to disk if enabled.
            - Prints progress and status messages if verbose is True.
            - sets self.trained to True, indicating the model has been trained.            
        Returns:
            None
        """       

        if patience is None:
            patience = torch.inf
            
        if saveModelEvery != torch.inf:
            if folder_temp is None:
                raise ValueError("If saveModelEvery is set to not inf, folder_temp must be set.")
            
        if max_epochs == torch.inf and patience == torch.inf:
            raise ValueError("Are you sure you want to train till the end of the universe?")
        
        
        if self.trained:
            if verbose:
                print("Model has already been trained or loaded.")
                if self.losses == None:
                    print("Restarting losses and lossesTest tensors.")
                        
                print("Last recorded bestLoss: " + str(self.bestLoss))
            
            if self.bestLoss is None:
                self.bestLoss = torch.inf
        
            
        
        self.losses = torch.zeros((len(train_dataloader), max_epochs)) 
        self.losses[self.losses == 0] = torch.nan
        
        if earlyStop_dataloader is None:
            pass
        else:
            self.lossesTest = torch.zeros(max_epochs)
            self.lossesTest[self.lossesTest == 0] = torch.nan


        
        print("Starting training on " + str(self.device) + " with " + str(self.world_size) + " processes.")
        if self.ddp:
            print(f"DDP is enabled")
            
            
            
        epoch = 0
        patienceCounter = 0
        sumTime = 0
        elapsedTime = -1

        while (epoch < max_epochs) and (patienceCounter < patience):
            
            if verbose:
                print("\n-----------------")
                

                print("GPU " + str(self.device) + ", Epoch: " + str(epoch) + "/" + str(max_epochs) + " @ " + str(datetime.datetime.now()) + "(+" + str(elapsedTime) + "s)")
                print("Current learning rate is: " + str(self.optimizer.param_groups[0]['lr']))
                if epoch > 0:
                    print(f"Train Loss: {self.losses[:, epoch-1].mean()}")
                    if earlyStop_dataloader is not None:
                        print(f"Validation current loss: {self.lossesTest[epoch-1]} / Best loss : {self.bestLoss}")
                    else:
                        print(f"Current loss: {self.losses[:, epoch-1].mean()} / Best loss : {self.bestLoss}")
                        
                start = time.time()
                
            
            batch_i = 0
            
            
            if self.ddp:
                train_dataloader.sampler.set_epoch(epoch)  # Ensure shuffling is consistent across epochs
            
            
            progress_bar = tqdm( train_dataloader, disable=(not self._is_main_device()) )
            for batch in progress_bar:
                if self.ddp:      
                    progress_bar.set_description(f"Running epoch {epoch} on {self.world_size} GPUs")
                else:
                    progress_bar.set_description(f"Running epoch {epoch} on device " + str(self.device))


                x = batch[0]
                y = batch[-1]
                extra = batch[1:-1]
                if len(extra) != 0:
                    forward_kwargs = { k: v for k, v in zip(self.additional_forward_args, extra) }
                else:
                    forward_kwargs = {}
                                    
                
                loss = self.train_iter(x = x,
                                       y = y, 
                                       loss_function = my_loss_function,
                                       **forward_kwargs)
                
                self.losses[batch_i, epoch] = loss
                                
                batch_i += 1
                


            if earlyStop_dataloader is None:
                loss_to_account = self.losses[:, epoch].mean()      
                if self.world_size > 1:
                    loss_to_account = loss_to_account.to(self.device)
                    all_reduce(loss_to_account, op=ReduceOp.AVG)
                    self.losses[:, epoch] = loss_to_account.cpu()
            else:
                self.__validate(my_loss_function, earlyStop_dataloader, epoch) # updates self.lossesTest[epoch]
                loss_to_account = self.lossesTest[epoch]
            
            
                
            if not (self.scheduler is None):
                self.scheduler_step(loss_to_account)



            if (loss_to_account < self.bestLoss):        
                patienceCounter = 0                
            else:
                patienceCounter += 1


            self.updateBestModelandLoss(loss_to_account) # already checks if loss is better than bestLoss and updates bestLoss


            if verbose:
                # print("Patience: " + str(patienceCounter) + "/" + str(patience))
                print("GPU " + str(self.device) + ", Patience: " + str(patienceCounter) + "/" + str(patience))
                elapsedTime = time.time() - start 
                sumTime += elapsedTime



            if ( (epoch + 1) % saveModelEvery ) == 0:
                self.save_state_dict( folder_temp + "/" + str(epoch) + "_" + datetime.datetime.now().strftime("%Y_%B_%d_%I:%M%p"),
                                    best = save_best_model, 
                                    verbose = verbose)

            if write_losses and ( (not self.ddp) or self._is_main_device() ):
                np.save(folder_temp + "/losses.npy", self.losses.to("cpu").numpy())
                
                if earlyStop_dataloader is not None:
                    np.save(folder_temp + "/lossesTest.npy", self.lossesTest.to("cpu").numpy()) 
                
                
                
            epoch += 1


        self.trained = True


        self.losses = self.losses[:, :epoch ]
        
        if earlyStop_dataloader is not None:        
            self.lossesTest = self.lossesTest[ :epoch ]
        
        
        
        if verbose:
            print("\n \n")
            print("Training finished after " + str(epoch) + " epochs.")
            print("Total training time: " + str(datetime.timedelta(seconds = sumTime)))
            print("Average time per epoch: " + str(sumTime / epoch))
            if earlyStop_dataloader is None:
                add_string = " (on training set): "
            else:    
                add_string = " (on validation set): "

            print("Best loss" + add_string  + str(self.bestLoss))
        
        
        
        if final_model_name is None:
            pass
        else:
            self.save_state_dict( final_model_name, best = save_best_model, verbose = verbose)

    def _is_main_device(self) -> bool:
        if self.ddp:
            return self.device == 0
        return True



    def __validate(self, my_loss, earlyStop_dataloader, epoch):
        
        # if self.device == 0: # Only rank 0 will perform validation                    
        # fixme: in its current form, each gpu is validating the entire dataset, which is not ideal.
        # this should be done on just 1 gpu, and then signaled to the other gpus.
        
        i_for_the_averaging = 0
        self.lossesTest[epoch] = 0
        for batch in earlyStop_dataloader: 
            x2d_early_stop_b = batch[0]
            y_early_stop_b = batch[-1]
            extra = batch[1:-1]
            if len(extra) != 0:
                forward_kwargs = { k: v for k, v in zip(self.additional_forward_args, extra) }
            else:
                forward_kwargs = {}
                    
            self.lossesTest[epoch] += self.test_iter(x2d_early_stop_b,
                                                    y = y_early_stop_b, 
                                                    loss_function = my_loss, 
                                                    verbose = False,
                                                    **forward_kwargs)                

            i_for_the_averaging += 1
                
        self.lossesTest[epoch] /= i_for_the_averaging




    def __validateDiffusion(self, my_loss, earlyStop_dataloader, noise_scheduler, epoch, validation_seed=None, 
    n_timesteps_eval=3):

        # if self.device == 0: # Only rank 0 will perform validation
        # fixme: in its current form, each gpu is validating the entire dataset, which is not ideal.
        # this should be done on just 1 gpu, and then signaled to the other gpus.

        # dedicated Generator: reseeding in each call to generate the same 
        # squence of noise/timesteps in all the epochs during the validation
        generator = None
        if validation_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(validation_seed)
        
        T = noise_scheduler.config['num_train_timesteps']
        bin_edges = torch.linspace(0, T, n_timesteps_eval + 1).long()

        i_for_the_averaging = 0
        self.lossesTest[epoch] = 0
        for clean_images, condition in earlyStop_dataloader: 
            for bin_i in range(n_timesteps_eval):
                low_t = bin_edges[bin_i].item()
                high_t = bin_edges[bin_i + 1].item()
                noise = torch.randn( clean_images.shape, generator=generator )
                timesteps = torch.randint( low = low_t, high = high_t, 
                                      size = (clean_images.shape[0], ), dtype=torch.long,
                                      generator=generator )
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
                # Conditional path based on model input
                if "encoder_hidden_states" in self.additional_forward_args:
                    noisy_images_and_condition = noisy_images
                    forward_kwargs = {"encoder_hidden_states": condition}
                else:
                    noisy_images_and_condition = torch.cat((noisy_images, condition), dim = 1)
                    forward_kwargs = {}
                

                self.lossesTest[epoch] += self.test_iter(noisy_images_and_condition,
                                                     y = noise, 
                                                     loss_function = my_loss, 
                                                     verbose = False,
                                                     timestep = timesteps,
                                                     **forward_kwargs)                

                i_for_the_averaging += 1
                
        self.lossesTest[epoch] /= i_for_the_averaging



     
        






    def trainDiffusionModel(self, 
                            train_dataloader,
                            noise_scheduler,
                            earlyStop_dataloader = None,
                            max_epochs = 1000, patience = 50, 
                            saveModelEvery = torch.inf,
                            write_losses = False,                   
                            folder_temp = None, 
                            final_model_name = None,
                            save_best = True,
                            verbose = True,
                            my_loss_function = torch.nn.MSELoss(reduction = 'mean'),
                            validation_seed = 0):                    
        """
        Trains the model for diffusion tasks using the provided training dataloader and noise scheduler, 
        with support for early stopping, learning rate scheduling, and periodic model checkpointing.
        This model is meant to be used with conditional diffusion models,
            where the input is a noisy image and the target is the clean image.

        Differences from trainModel:
            - Designed for diffusion models, expects batches where the first channel is the target (clean images)
              and the remaining channels are conditioning information.
            - Requires a noise_scheduler to generate noisy images and timesteps.
            - Uses MSE loss between predicted and true noise. This is hardcoded since diffusion models work with MSE
            
        Args:
            train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset. Must contain two tensors in each batch:
                - [0]: target (clean images)
                - [1]: conditioning            
            max_epochs (int, optional): Maximum number of epochs to train. Defaults to 1000.
            patience (int, optional): Number of epochs to wait for improvement in validation loss before stopping early. Defaults to 50.
            saveModelEvery (int or float, optional): Frequency (in epochs) to save model checkpoints. If set to torch.inf, disables periodic saving. Defaults to torch.inf.
            write_losses (bool, optional): Whether to save the training and validation loss arrays to disk after each epoch. Defaults to False.
            folder_temp (str, optional): Directory to save model checkpoints and loss arrays. Required if saveModelEvery is not torch.inf.
            final_model_name (str, optional): Path to save the final best model after training. If None, the model is not saved at the end. Defaults to None.
            verbose (bool, optional): Whether to print progress and status messages during training. Defaults to True.
            validation_seed (int, optional): Seed for the RNG used to generate noise and timesteps during validation. 
            Fixed so that the same (noise, timestep) sequence is used across all epochs, making lossTest comparable epoch-to-epoch. Set to None to disable. Defaults to 0.
        Raises:
            ValueError: If saveModelEvery is not torch.inf and folder_temp is not provided.
            ValueError: If both max_epochs and patience are set to torch.inf.
        Side Effects:
            - Updates self.losses and self.lossesTest with training and validation losses.
            - Updates self.bestLoss and self.bestmodelStateDict with the best validation loss and corresponding model parameters.
            - Saves model checkpoints and loss arrays to disk if enabled.
            - Prints progress and status messages if verbose is True.
            - sets self.trained to True, indicating the model has been trained.
        Returns:
            None
        """                     
            
        if patience is None:
            patience = torch.inf
            
        if saveModelEvery != torch.inf:
            if folder_temp is None:
                raise ValueError("If saveModelEvery is set to not inf, folder_temp must be set.")
            
        if max_epochs == torch.inf and patience == torch.inf:
            raise ValueError("Are you sure you want to train till the end of the universe?")
        
        
        if self.trained:
            if verbose:
                print("Model has already been trained or loaded.")
                if self.losses == None:
                    print("Restarting losses and lossesTest tensors.")
                        
                print("Last recorded bestLoss: " + str(self.bestLoss))

            if self.bestLoss is None:
                self.bestLoss = torch.inf


            
        
        self.losses = torch.zeros((len(train_dataloader), max_epochs)) 
        self.losses[self.losses == 0] = torch.nan
        
        if earlyStop_dataloader is None:
            pass
        else:
            self.lossesTest = torch.zeros(max_epochs)
            self.lossesTest[self.lossesTest == 0] = torch.nan
            
            
            
        print("Starting training on " + str(self.device) + " with " + str(self.world_size) + " processes.")
        if self.ddp:
            print(f"DDP is enabled")
            
            
            
        epoch = 0
        patienceCounter = 0
        sumTime = 0
        elapsedTime = -1

        while (epoch < max_epochs) and (patienceCounter < patience):
            
            if verbose:
                print("\n-----------------")
                

                print("GPU " + str(self.device) + ", Epoch: " + str(epoch) + "/" + str(max_epochs) + " @ " + str(datetime.datetime.now()) + "(+" + str(elapsedTime) + "s)")
                print("Current learning rate is: " + str(self.optimizer.param_groups[0]['lr']))
                if epoch > 0:
                    if earlyStop_dataloader is not None:
                        print(f"Train Loss: {self.losses[:, epoch-1].mean()}")
                        print(f"Validation current loss: {self.lossesTest[epoch-1]} / Best loss : {self.bestLoss}")
                    else:
                        print(f"Train current loss: {self.losses[:, epoch-1].mean()} / Best loss: {self.bestLoss}")
                start = time.time()
                

            if self.ddp:
                train_dataloader.sampler.set_epoch(epoch)  # Ensure shuffling is consistent across epochs
            # Note from the warning in DistributedSampler
            # In distributed mode, calling the set_epoch method at the beginning of each epoch before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be always used.
            
            
            batch_i = 0
            progress_bar = tqdm( train_dataloader, disable = (not self._is_main_device()) )
            for clean_images, condition in progress_bar:          

                noise = torch.randn( clean_images.shape )

                timesteps = torch.randint( low = 0, high = noise_scheduler.config['num_train_timesteps'], 
                        size = (clean_images.shape[0], ), dtype=torch.long)
            
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                # If model expects encoder_hidden_states (UNet2DConditionModel)
                # pass condition as a kwarg; otherwise, concatenate
                if "encoder_hidden_states" in self.additional_forward_args:
                    noisy_images_and_condition = noisy_images
                    forward_kwargs = {"encoder_hidden_states": condition}
                else:
                    noisy_images_and_condition = torch.cat((noisy_images, condition), dim = 1)
                    forward_kwargs = {}

                
                self.losses[batch_i, epoch] = self.train_iter(x = noisy_images_and_condition,
                                                              y = noise,
                                                              timestep = timesteps,
                                                              loss_function = my_loss_function,
                                                              **forward_kwargs
                                                              )

                self._update_ema()

                if self.ddp:
                    progress_bar.set_description(f"Running epoch {epoch} on {self.world_size} GPUs. Current Loss: {self.losses[batch_i, epoch]}")
                else:
                    progress_bar.set_description(f"Running epoch {epoch} on device " + str(self.device))

                                
                batch_i += 1
                                       

            
            if earlyStop_dataloader is None:
                loss_to_account = self.losses[:, epoch].mean()
                if self.world_size > 1:
                    loss_to_account = loss_to_account.to(self.device)
                    all_reduce(loss_to_account, op=ReduceOp.AVG)
                    self.losses[:, epoch] = loss_to_account.cpu()
            else: # note: as of now, all gpus will perform the same validation. This is quick and probably fine since communication *may* be slower
                self.__validateDiffusion(my_loss_function, earlyStop_dataloader, noise_scheduler, epoch,
                validation_seed=validation_seed) # updates self.lossesTest[epoch]
                loss_to_account = self.lossesTest[epoch]
            
            

            if not (self.scheduler is None):
                self.scheduler_step(loss_to_account)


            if (loss_to_account < self.bestLoss):        
                patienceCounter = 0                
            else:
                patienceCounter += 1


            self.updateBestModelandLoss(loss_to_account) # already checks if loss is better than bestLoss and updates bestLoss


            if verbose:    
                # print("Patience: " + str(patienceCounter) + "/" + str(patience))
                print("GPU " + str(self.device) + ", Patience: " + str(patienceCounter) + "/" + str(patience))
                elapsedTime = time.time() - start 
                sumTime += elapsedTime



            if ( (epoch + 1) % saveModelEvery ) == 0:
                self.save_state_dict( folder_temp + "/" + str(epoch) + "_" + datetime.datetime.now().strftime("%Y_%B_%d_%I:%M%p"),
                                    best = save_best, 
                                    verbose = verbose)

            if write_losses and ( (not self.ddp) or self._is_main_device() ):  # fixme this is just the losses for one device. change losses to loss_to_account
                np.save(folder_temp + "/losses.npy", self.losses.to("cpu").numpy())
                if earlyStop_dataloader is not None:
                    np.save( folder_temp + "/lossesTest.npy", self.lossesTest.to("cpu").numpy() )
                
                
            epoch += 1


        self.trained = True


        self.losses = self.losses[:, :epoch ]
        if earlyStop_dataloader is not None:        
            self.lossesTest = self.lossesTest[ :epoch ]
                
        
        
        if verbose:
            print("\n \n")
            print("Training finished after " + str(epoch) + " epochs.")
            print("Total training time: " + str(datetime.timedelta(seconds = sumTime)))
            print("Average time per epoch: " + str(sumTime / epoch))
            
            if earlyStop_dataloader is not None:
                add_string = " (on validation set): "
            else:
                add_string = " (on training set): "
            print("Best loss" + add_string  + str(self.bestLoss))



        if final_model_name is None:
            pass
        else:
            self.save_state_dict( final_model_name, best = save_best, verbose = verbose)
            if self.ema_state_dict is not None:
                self.save_ema_state_dict( final_model_name + "_ema", best = save_best, verbose = verbose)





    def simulateDiffusion(self, dataloader, noise_scheduler, sub_batch_size = None, transformSimulation = None, file_name = None,
                          verbose = True, **step_kwargs):
        """
        Simulates the diffusion process over a dataset using a noise scheduler and saves or returns the results.
        Note: If launched on parallel, this method requires backend = "gloo", since it pushes the results to cpus
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader providing (sample, conditioning) pairs for simulation.
                Assumes sample is pregenerated noise (typically normally distributed noise)
                Will generate samples based on this
            noise_scheduler: An object that provides the diffusion timesteps and a `step` method to update samples.
            sub_batch_size (int, optional): If provided, the dataloader is split into sub-batches of this size for processing.
            transformSimulation (callable, optional): A function to apply to the simulation results before saving or returning.
            file_name (str, optional): If provided, the simulation results are saved to this file (as a NumPy .npy file).
                                        If None, the simulation results are returned as a NumPy array.
        Returns:
            np.ndarray or None: The simulated data as a NumPy array if `file_name` is None; otherwise, saves the data to disk and returns None.
        Notes:
            - If running in a distributed setting (`self.world_size > 1`), the file name is suffixed with the device rank.
            - The method assumes that `self.predict` and `noise_scheduler.step` are implemented and compatible with the data shapes.
            - The simulation is performed by iteratively applying the model's prediction and the noise scheduler's step for each timestep.
        """
        
        
        for thingys in dataloader: # to get sample shape
            sample = thingys[0]
            break
        
        if len(thingys) == 3:
            has_indexing = True
        else:
            has_indexing = False
            if self.world_size > 1:
                print("Warning: no index was found and this seems to be a distributed simulation. It will not be possible to remove potential duplicates and beware of the order.")
        
        
        if verbose:
            if self._is_main_device():
                print("Simulating on " + str(self.world_size) + " devices")

        uses_subdataloader = False
        if sub_batch_size is not None:
            if sample.shape[0] > sub_batch_size:
                uses_subdataloader = True

                

        if uses_subdataloader:            
            progress_bar = dataloader  
        else:
            if verbose: 
                progress_bar = tqdm(dataloader, disable = (not self._is_main_device()) )  
            else:
                progress_bar = dataloader


        
        use_encoder_conditioning = "encoder_hidden_states" in self.additional_forward_args



        simulation = torch.zeros((0, *sample.shape[1:]))
        if has_indexing:
            index = torch.zeros((0))                                     
        for thingys in progress_bar:
            

            
            sample = thingys[0] 
            conditioning = thingys[1]  
            if uses_subdataloader:
                if verbose:
                    sub_dataloader = tqdm(torch.utils.data.DataLoader( torch.utils.data.TensorDataset(sample, conditioning),
                                                                    batch_size = sub_batch_size,
                                                                    shuffle = False), 
                                        disable = (not self._is_main_device())
                                        )
                else:
                    sub_dataloader = torch.utils.data.DataLoader( torch.utils.data.TensorDataset(sample, conditioning),
                                                                batch_size = sub_batch_size,
                                                                shuffle = False)
            else:
                sub_dataloader = [(sample, conditioning)]
            
            
            training_mode = self.model.training
            self.model.eval()            
            with torch.inference_mode():                
                for sample, conditioning in sub_dataloader:    
                    batch_size, channels, height, width = sample.shape
                    conditioning = conditioning.to(self.device)
                    
                    
                    if not use_encoder_conditioning:
                        sample_and_condition = torch.empty( (batch_size, channels + conditioning.shape[1], height, width),
                                                            device=self.device, dtype=sample.dtype)
                        sample_and_condition[:, channels:, ... ].copy_(conditioning)


                    # rebuild a clean scheduler so step_index/model_outputs start fresh
                    noise_scheduler.set_timesteps(len(noise_scheduler.timesteps), device=self.device)

                    sample = sample * noise_scheduler.init_noise_sigma # initialize the latent with the scheduler sigma
                    sample = sample.to(self.device)    # makes sure it doesn't go back and forth between cpu and gpu, since self.predict will move it to self.device but then put it back to original device
                    
                    
                    
                    for timestep_ in noise_scheduler.timesteps:
                        #sample_and_condition = torch.cat((noise_scheduler.scale_model_input(sample, timestep_), conditioning), dim = 1) # concatenate the background prediction to the dataset
                        scaled_input = noise_scheduler.scale_model_input(sample, timestep_)
                        if use_encoder_conditioning:
                            residual = self.model(scaled_input, timestep=timestep_, encoder_hidden_states=conditioning)
                        else:
                            sample_and_condition[:, :channels, ... ].copy_(scaled_input)
                            residual = self.model(sample_and_condition, timestep=timestep_)

                        sample = noise_scheduler.step(model_output = residual, timestep = timestep_, sample = sample, **step_kwargs).prev_sample
                            
                    simulation = torch.concatenate((simulation, sample.cpu()), dim=0)

            self.model.train(mode=training_mode)

            if has_indexing:
                index = torch.concatenate((index, thingys[2]))
                
            
        
        if self.world_size > 1:
            # Gather the simulation results from all devices
            # Note: This assumes that the simulation is on the CPU, so necessitates backend = "gloo"
            simulations_on_all_devices = [torch.zeros(size = simulation.shape) for _ in range(self.world_size)] if self._is_main_device() else None                
            gather(simulation, simulations_on_all_devices, dst = 0)
            
            if has_indexing:
                indices_on_all_devices = [torch.zeros(size = index.shape) for _ in range(self.world_size)] if self._is_main_device() else None
                gather(index, indices_on_all_devices, dst = 0)
        
        
        if self._is_main_device():
            if self.world_size > 1:            
                simulation = torch.cat(simulations_on_all_devices, dim=0)
                if has_indexing:
                    index = torch.cat(indices_on_all_devices, dim=0)
                    
                    # Sorts the simulation and removes duplicates:
                    simulation = simulation[np.unique(index, return_index = True)[1]]
        
        
            if transformSimulation is not None:    
                simulation = transformSimulation(simulation)                


            simulation = simulation.numpy()  # Convert to numpy array for saving
            
            if file_name is None:
                return simulation
            else:                
                np.save(file_name, simulation)


    """     
    # if Im gonna do this in parallel they should join the main process, the next is generated by copilot:
    def simulate_diffusion_parallel(self, dataloader, noise_scheduler, dataset_min, dataset_max, file_name):

        if not self.ddp:
            raise ValueError("This method is only available when using DDP.")
        
        # Each process will handle its own part of the data
        simulation = self.simulate_diffusion(dataloader, noise_scheduler, dataset_min, dataset_max, file_name)
        
        # Gather results from all processes
        gathered_simulations = [torch.zeros_like(simulation) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered_simulations, simulation)
        
        # Concatenate results from all processes
        final_simulation = torch.cat(gathered_simulations, dim=0)
        
        # Save the final simulation
        np.save(file_name, final_simulation.cpu().numpy())
        
        return final_simulation.cpu().numpy()
    """

    # Flow matching implementation: code generated using Claude code
    def __validateFlowMatching(self, my_loss, earlyStop_dataloader, epoch, validation_seed=None, time_scale=1000.0):
        generator = None
        if validation_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(validation_seed)

        i_for_the_averaging = 0
        self.lossesTest[epoch] = 0
        for clean_images, condition in earlyStop_dataloader:
            x0 = torch.randn(clean_images.shape, generator=generator)
            t = torch.rand(clean_images.shape[0], generator=generator)

            t_broadcast = t.view(-1, *([1] * (clean_images.dim() - 1)))
            xt = (1 - t_broadcast) * x0 + t_broadcast * clean_images
            velocity_target = clean_images - x0

            timestep_input = t * time_scale

            if "encoder_hidden_states" in self.additional_forward_args:
                xt_and_condition = xt
                forward_kwargs = {"encoder_hidden_states": condition}
            else:
                xt_and_condition = torch.cat((xt, condition), dim=1)
                forward_kwargs = {}

            self.lossesTest[epoch] += self.test_iter(xt_and_condition,
                                                     y=velocity_target,
                                                     loss_function=my_loss,
                                                     verbose=False,
                                                     timestep=timestep_input,
                                                     **forward_kwargs)

            i_for_the_averaging += 1

        self.lossesTest[epoch] /= i_for_the_averaging

    def trainFlowMatchingModel(self,
                                train_dataloader,
                                earlyStop_dataloader=None,
                                max_epochs=1000, patience=50,
                                saveModelEvery=torch.inf,
                                write_losses=False,
                                folder_temp=None,
                                final_model_name=None,
                                save_best=True,
                                verbose=True,
                                my_loss_function=torch.nn.MSELoss(reduction='mean'),
                                validation_seed=0,
                                time_scale=1000.0):
        """
        Trains the model with conditional flow matching instead of DDPM-style denoising diffusion.
        Same batch format and DDP/EMA/early-stopping machinery as trainDiffusionModel, but:
            - No noise_scheduler needed: samples t ~ Uniform(0,1) directly, no discrete schedule.
            - Target is the (constant) velocity x1 - x0 along the linear path x_t = (1-t)x0 + t*x1,
            instead of the injected noise.
            - time_scale rescales the continuous t in [0,1] before feeding it to the model's timestep
            embedding, which is designed for the ~[0, num_train_timesteps) range (see conversation notes /
            docstring above the method definition in the source for the full rationale). Must match the
            time_scale used later at sampling time in simulateFlowMatching.
        Args: same as trainDiffusionModel, minus noise_scheduler, plus time_scale (float, optional):
            Scale factor applied to t before it's passed to the model's timestep embedding. Defaults to 1000.0.
        """

        if patience is None:
            patience = torch.inf

        if saveModelEvery != torch.inf:
            if folder_temp is None:
                raise ValueError("If saveModelEvery is set to not inf, folder_temp must be set.")

        if max_epochs == torch.inf and patience == torch.inf:
            raise ValueError("Are you sure you want to train till the end of the universe?")

        if self.trained:
            if verbose:
                print("Model has already been trained or loaded.")
                if self.losses == None:
                    print("Restarting losses and lossesTest tensors.")
                print("Last recorded bestLoss: " + str(self.bestLoss))
            if self.bestLoss is None:
                self.bestLoss = torch.inf

        self.losses = torch.zeros((len(train_dataloader), max_epochs))
        self.losses[self.losses == 0] = torch.nan

        if earlyStop_dataloader is None:
            pass
        else:
            self.lossesTest = torch.zeros(max_epochs)
            self.lossesTest[self.lossesTest == 0] = torch.nan

        print("Starting training on " + str(self.device) + " with " + str(self.world_size) + " processes.")
        if self.ddp:
            print(f"DDP is enabled")

        epoch = 0
        patienceCounter = 0
        sumTime = 0
        elapsedTime = -1

        while (epoch < max_epochs) and (patienceCounter < patience):

            if verbose:
                print("\n-----------------")
                print("GPU " + str(self.device) + ", Epoch: " + str(epoch) + "/" + str(max_epochs) + " @ " + str(datetime.datetime.now()) + "(+" + str(elapsedTime) + "s)")
                print("Current learning rate is: " + str(self.optimizer.param_groups[0]['lr']))
                if epoch > 0:
                    if earlyStop_dataloader is not None:
                        print(f"Train Loss: {self.losses[:, epoch-1].mean()}")
                        print(f"Validation current loss: {self.lossesTest[epoch-1]} / Best loss : {self.bestLoss}")
                    else:
                        print(f"Train current loss: {self.losses[:, epoch-1].mean()} / Best loss: {self.bestLoss}")
                start = time.time()

            if self.ddp:
                train_dataloader.sampler.set_epoch(epoch)

            batch_i = 0
            progress_bar = tqdm(train_dataloader, disable=(not self._is_main_device()))
            for clean_images, condition in progress_bar:

                x0 = torch.randn(clean_images.shape)
                t = torch.rand(clean_images.shape[0])

                t_broadcast = t.view(-1, *([1] * (clean_images.dim() - 1)))
                xt = (1 - t_broadcast) * x0 + t_broadcast * clean_images
                velocity_target = clean_images - x0

                timestep_input = t * time_scale

                if "encoder_hidden_states" in self.additional_forward_args:
                    xt_and_condition = xt
                    forward_kwargs = {"encoder_hidden_states": condition}
                else:
                    xt_and_condition = torch.cat((xt, condition), dim=1)
                    forward_kwargs = {}

                self.losses[batch_i, epoch] = self.train_iter(x=xt_and_condition,
                                                              y=velocity_target,
                                                              timestep=timestep_input,
                                                              loss_function=my_loss_function,
                                                              **forward_kwargs)

                self._update_ema()

                if self.ddp:
                    progress_bar.set_description(f"Running epoch {epoch} on {self.world_size} GPUs. Current Loss: {self.losses[batch_i, epoch]}")
                else:
                    progress_bar.set_description(f"Running epoch {epoch} on device " + str(self.device))

                batch_i += 1

            if earlyStop_dataloader is None:
                loss_to_account = self.losses[:, epoch].mean()
                if self.world_size > 1:
                    loss_to_account = loss_to_account.to(self.device)
                    all_reduce(loss_to_account, op=ReduceOp.AVG)
                    self.losses[:, epoch] = loss_to_account.cpu()
            else:
                self.__validateFlowMatching(my_loss_function, earlyStop_dataloader, epoch,
                                            validation_seed=validation_seed, time_scale=time_scale)
                loss_to_account = self.lossesTest[epoch]

            if not (self.scheduler is None):
                self.scheduler_step(loss_to_account)

            if (loss_to_account < self.bestLoss):
                patienceCounter = 0
            else:
                patienceCounter += 1

            self.updateBestModelandLoss(loss_to_account)

            if verbose:
                print("GPU " + str(self.device) + ", Patience: " + str(patienceCounter) + "/" + str(patience))
                elapsedTime = time.time() - start
                sumTime += elapsedTime

            if ((epoch + 1) % saveModelEvery) == 0:
                self.save_state_dict(folder_temp + "/" + str(epoch) + "_" + datetime.datetime.now().strftime("%Y_%B_%d_%I:%M%p"),
                                    best=save_best, verbose=verbose)

            if write_losses and ((not self.ddp) or self._is_main_device()):
                np.save(folder_temp + "/losses.npy", self.losses.to("cpu").numpy())
                if earlyStop_dataloader is not None:
                    np.save( folder_temp + "/lossesTest.npy", self.lossesTest.to("cpu").numpy() )

            epoch += 1

        self.trained = True
        self.losses = self.losses[:, :epoch]
        if earlyStop_dataloader is not None:
            self.lossesTest = self.lossesTest[:epoch]

        if verbose:
            print("\n \n")
            print("Training finished after " + str(epoch) + " epochs.")
            print("Total training time: " + str(datetime.timedelta(seconds=sumTime)))
            print("Average time per epoch: " + str(sumTime / epoch))
            add_string = " (on validation set): " if earlyStop_dataloader is not None else " (on training set): "
            print("Best loss" + add_string + str(self.bestLoss))

        if final_model_name is None:
            pass
        else:
            self.save_state_dict(final_model_name, best=save_best, verbose=verbose)
            if self.ema_state_dict is not None:
                self.save_ema_state_dict(final_model_name + "_ema", best=save_best, verbose=verbose)



    def simulateFlowMatching(self, dataloader, n_steps=50, solver="euler",
                            sub_batch_size=None, transformSimulation=None,
                            file_name=None,
                            time_scale=1000.0, verbose=True):
        """
        Generates samples by integrating the learned velocity field from t=0 (noise) to t=1 (data),
        using either Euler (1 model call per step) or Heun / RK2 (2 model calls per step, more accurate
        per step, useful for keeping n_steps low). Mirrors simulateDiffusion's dataloader/DDP/gather/
        sub_batch_size/transformSimulation/file_name interface, but with no noise_scheduler: n_steps and
        solver take its place. time_scale must match the value used in trainFlowMatchingModel.
        Args:
            dataloader: yields (sample, conditioning[, index]); sample is x0 ~ N(0,I), same convention
                as simulateDiffusion (no init_noise_sigma scaling here -- flow matching starts from raw
                standard normal noise, not a scheduler-specific sigma).
            n_steps (int, optional): number of ODE integration steps. Flow matching typically needs far
                fewer than DDPM's ~1000 (try 20-50 to start). Defaults to 50.
            solver (str, optional): "euler" or "heun". Defaults to "euler".
        Returns:
            np.ndarray or None: as simulateDiffusion.
        """

        if solver not in ("euler", "heun"):
            raise ValueError("solver must be 'euler' or 'heun'")

        for thingys in dataloader:
            sample = thingys[0]
            break

        if len(thingys) == 3:
            has_indexing = True
        else:
            has_indexing = False
            if self.world_size > 1:
                print("Warning: no index was found and this seems to be a distributed simulation. It will not be possible to remove potential duplicates and beware of the order.")

        if verbose:
            if self._is_main_device():
                print("Simulating on " + str(self.world_size) + " devices")

        uses_subdataloader = False
        if sub_batch_size is not None:
            if sample.shape[0] > sub_batch_size:
                uses_subdataloader = True

        if uses_subdataloader:
            progress_bar = dataloader
        else:
            if verbose:
                progress_bar = tqdm(dataloader, disable=(not self._is_main_device()))
            else:
                progress_bar = dataloader

        use_encoder_conditioning = "encoder_hidden_states" in self.additional_forward_args

        dt = 1.0 / n_steps

        simulation = torch.zeros((0, *sample.shape[1:]))
        if has_indexing:
            index = torch.zeros((0))
        for thingys in progress_bar:

            sample = thingys[0]
            conditioning = thingys[1]
            if uses_subdataloader:
                if verbose:
                    sub_dataloader = tqdm(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(sample, conditioning),
                                                                    batch_size=sub_batch_size, shuffle=False),
                                        disable=(not self._is_main_device()))
                else:
                    sub_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(sample, conditioning),
                                                                batch_size=sub_batch_size, shuffle=False)
            else:
                sub_dataloader = [(sample, conditioning)]

            training_mode = self.model.training
            self.model.eval()
            with torch.inference_mode():
                for sample, conditioning in sub_dataloader:
                    batch_size, channels, height, width = sample.shape
                    conditioning = conditioning.to(self.device)

                    if not use_encoder_conditioning:
                        sample_and_condition = torch.empty((batch_size, channels + conditioning.shape[1], height, width),
                                                            device=self.device, dtype=sample.dtype)
                        sample_and_condition[:, channels:, ...].copy_(conditioning)

                    x = sample.to(self.device)  # x0 ~ N(0,I), no scheduler-specific rescaling needed here

                    for step in range(n_steps):
                        t_val = step * dt
                        t_batch = torch.full((batch_size,), t_val, device=self.device) * time_scale

                        if use_encoder_conditioning:
                            v1 = self.model(x, timestep=t_batch, encoder_hidden_states=conditioning)
                        else:
                            sample_and_condition[:, :channels, ...].copy_(x)
                            v1 = self.model(sample_and_condition, timestep=t_batch)

                        if solver == "euler":
                            x = x + dt * v1
                        else:  # heun
                            x_euler = x + dt * v1
                            t_next_batch = torch.full((batch_size,), t_val + dt, device=self.device) * time_scale

                            if use_encoder_conditioning:
                                v2 = self.model(x_euler, timestep=t_next_batch, encoder_hidden_states=conditioning)
                            else:
                                sample_and_condition[:, :channels, ...].copy_(x_euler)
                                v2 = self.model(sample_and_condition, timestep=t_next_batch)

                            x = x + dt * 0.5 * (v1 + v2)

                    simulation = torch.concatenate((simulation, x.cpu()), dim=0)

            self.model.train(mode=training_mode)

            if has_indexing:
                index = torch.concatenate((index, thingys[2]))

        if self.world_size > 1:
            simulations_on_all_devices = [torch.zeros(size=simulation.shape) for _ in range(self.world_size)] if self._is_main_device() else None
            gather(simulation, simulations_on_all_devices, dst=0)

            if has_indexing:
                indices_on_all_devices = [torch.zeros(size=index.shape) for _ in range(self.world_size)] if self._is_main_device() else None
                gather(index, indices_on_all_devices, dst=0)

        if self._is_main_device():
            if self.world_size > 1:
                simulation = torch.cat(simulations_on_all_devices, dim=0)
                if has_indexing:
                    index = torch.cat(indices_on_all_devices, dim=0)
                    simulation = simulation[np.unique(index, return_index=True)[1]]

            if transformSimulation is not None:
                simulation = transformSimulation(simulation)

            simulation = simulation.numpy()

            if file_name is None:
                return simulation
            else:
                np.save(file_name, simulation)
