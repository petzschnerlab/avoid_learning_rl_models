from helpers.tasks import AvoidanceLearningTask
from helpers.rl_models import QLearning

class RLPipeline:
        def __init__(self, task, model, trial_design):

            #Get parameters
            self.trial_design = trial_design
            self.task = task
            self.task.initiate_model(model)

        def simulate(self):

            #Run simulation and computations
            self.task.run_experiment(self.trial_design)
            self.task.rl_model.run_computations()

            return self.task.rl_model

if __name__ == "__main__":

    #Initialize task, model, and task design
    task = AvoidanceLearningTask()
    model = QLearning(factual_lr=0.1, counterfactual_lr=0.05, temperature=0.1)
    task_design = {'learning_phase': {'number_of_trials': 24, 'number_of_blocks': 4},
                    'transfer_phase': {'times_repeated': 4}}
            
    #Initialize pipeline
    q_learning = RLPipeline(task, model, task_design).simulate()

    #Finalize and view model
    q_learning.plot_model()
    
    #Debug stop
    print()