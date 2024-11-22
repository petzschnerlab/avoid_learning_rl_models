from helpers.task import AvoidanceLearningTask
from helpers.rl_models import QLearning

if __name__ == "__main__":
    
    #Initialize Q-Learning model
    q_learning = QLearning(factual_lr=0.1, 
                           counterfactual_lr=0.1, 
                           temperature=0.1)
    
    #Initialize task
    al_task = AvoidanceLearningTask(q_learning)

    #Run experiment
    al_task.run_experiment()
    q_learning = al_task.rl_model

    #Plot Q-values
    q_learning.plot_q_values()

    #Debug stop
    print()