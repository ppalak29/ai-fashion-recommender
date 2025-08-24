import math
import random

class TrainableNeuralNetwork:
    
    def __init__(self):
        # Initialize weights RANDOMLY
        # 2 inputs, 3 layer 1 neurons
        self.input_to_hidden_weights = [
            [random.uniform(-1, 1), random.uniform(-1, 1)],  # Random weights for hidden neuron 1
            [random.uniform(-1, 1), random.uniform(-1, 1)],  # Random weights for hidden neuron 2  
            [random.uniform(-1, 1), random.uniform(-1, 1)]   # Random weights for hidden neuron 3
        ]
        
        self.hidden_biases = [
            random.uniform(-1, 1),  # Random bias for each hidden neuron
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ]
        
        self.hidden_to_output_weights = [
            random.uniform(-1, 1),  # Random weights from hidden to output
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ]
        
        self.output_bias = random.uniform(-1, 1)
        
        # Learning rate - how big steps to take when adjusting weights
        self.learning_rate = 0.5
    
    def sigmoid(self, x): #activation function
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0 if x < 0 else 1
    
    def sigmoid_derivative(self, x):
        """
        This is needed for backpropagation (learning).
        Formula: sigmoid(x) * (1 - sigmoid(x))
        """
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def forward_pass(self, temperature, formality):
        
        # Normalize inputs
        temp_normalized = temperature / 100.0
        formal_normalized = formality / 10.0
        inputs = [temp_normalized, formal_normalized]
        
        # Calculate hidden layer
        hidden_weighted_sums = []
        hidden_activations = []
        
        for i in range(3):
            weighted_sum = 0.0
            for j in range(len(inputs)):
                weighted_sum += inputs[j] * self.input_to_hidden_weights[i][j]
            weighted_sum += self.hidden_biases[i]
            
            activation = self.sigmoid(weighted_sum)
            
            hidden_weighted_sums.append(weighted_sum)  # Store for backpropagation
            hidden_activations.append(activation)
        
        # Calculate output layer
        output_weighted_sum = 0.0
        for i in range(len(hidden_activations)):
            output_weighted_sum += hidden_activations[i] * self.hidden_to_output_weights[i]
        output_weighted_sum += self.output_bias
        
        output = self.sigmoid(output_weighted_sum)
        
        # Return everything we need for learning
        return {
            'inputs': inputs,
            'hidden_weighted_sums': hidden_weighted_sums,
            'hidden_activations': hidden_activations,
            'output_weighted_sum': output_weighted_sum,
            'output': output
        }
    
    def train_one_example(self, temperature, formality, correct_answer):
        """
        Args:
            temperature: Input temperature
            formality: Input formality  
            correct_answer: What the output SHOULD be (0-1)
        """
        
        # Step 1: Forward pass - get the network's current guess
        forward_result = self.forward_pass(temperature, formality)
        network_guess = forward_result['output']
        
        # Step 2: Calculate the error
        error = correct_answer - network_guess
        print(f"Network guessed {network_guess:.3f}, correct answer was {correct_answer:.3f}, error = {error:.3f}")
        
        # Step 3: BACKPROPAGATION - work backwards to fix the weights
        
        output_error = error * self.sigmoid_derivative(forward_result['output_weighted_sum'])
        
        # Rule: new_weight = old_weight + learning_rate * output_error * hidden_activation
        for i in range(len(self.hidden_to_output_weights)):
            # TODO: Update self.hidden_to_output_weights[i]
            self.hidden_to_output_weights[i] += self.learning_rate * output_error * forward_result['hidden_activations'][i]
            pass
        
        self.output_bias += self.learning_rate * output_error
        
        # Hidden layer errors (how much each hidden neuron contributed to the error)
        hidden_errors = []
        for i in range(3):
            # Formula: output_error * weight_from_hidden_i_to_output * sigmoid_derivative_of_hidden_i
            hidden_error = 0.0  
            hidden_error = output_error * self.hidden_to_output_weights[i] * self.sigmoid_derivative(forward_result['hidden_weighted_sums'][i])
            hidden_errors.append(hidden_error)
        
        # Update hidden layer weights
        for i in range(3):
            for j in range(2):
                # Rule: new_weight = old_weight + learning_rate * hidden_error * input
                self.input_to_hidden_weights[i][j] += self.learning_rate * hidden_errors[i] * forward_result['inputs'][j]
                pass
            
            self.hidden_biases[i] += self.learning_rate * hidden_errors[i]
    
    def train(self, training_data, epochs=1000):
        """
        Args:
            training_data: List of (temperature, formality, correct_output) tuples
            epochs: How many times to go through all the training data
        """
                
        for epoch in range(epochs):
            total_error = 0
            
            random.shuffle(training_data)
            
            for temperature, formality, correct_answer in training_data:
                forward_result = self.forward_pass(temperature, formality)
                network_guess = forward_result['output']
                error = abs(correct_answer - network_guess)
                total_error += error
                
                self.train_one_example(temperature, formality, correct_answer)
            
            if epoch % 100 == 0:
                avg_error = total_error / len(training_data)
                print(f"Epoch {epoch}: Average error = {avg_error:.4f}")
        
        print("Training complete!")
    
    def predict(self, temperature, formality):
        """Make a prediction (same as before)"""
        result = self.forward_pass(temperature, formality)
        output = result['output']
        
        if output > 0.7:
            recommendation = "FORMAL BUSINESS ATTIRE"
        elif output > 0.4:
            recommendation = "BUSINESS CASUAL"
        else:
            recommendation = "CASUAL WEAR"
            
        return {
            'score': output,
            'recommendation': recommendation
        }


def create_training_data():
    
    training_data = [
        # (temperature, formality, correct_output_score)
        # Cold + Formal = High formality score
        (20, 9, 0.9),   # Very cold wedding
        (30, 8, 0.8),   # Cold formal event
        (25, 7, 0.75),  # Cold business meeting
        
        # Hot + Casual = Low formality score  
        (85, 2, 0.1),   # Hot beach day
        (90, 1, 0.05),  # Very hot casual
        (80, 3, 0.2),   # Warm casual hangout
        
        # Mixed scenarios
        (60, 5, 0.5),   # Moderate everything
        (70, 6, 0.6),   # Warm business casual
        (40, 4, 0.4),   # Cool casual
        
        # Edge cases
        (90, 9, 0.8),   # Hot but very formal (still need formal wear!)
        (20, 2, 0.3),   # Cold but casual (layers but casual)
    ]
    
    return training_data


def test_learning():
    
    network = TrainableNeuralNetwork()
    
    print("--- BEFORE TRAINING ---")
    test_cases = [(30, 8), (75, 3), (50, 6)]
    for temp, formality in test_cases:
        result = network.predict(temp, formality)
        print(f"Temp {temp}°F, Formality {formality}: {result['recommendation']} (score: {result['score']:.3f})")
    
    print("\n--- TRAINING ---")
    training_data = create_training_data()
    network.train(training_data, epochs=500)
    
    print("\n--- AFTER TRAINING ---")
    for temp, formality in test_cases:
        result = network.predict(temp, formality)
        print(f"Temp {temp}°F, Formality {formality}: {result['recommendation']} (score: {result['score']:.3f})")

if __name__ == "__main__":
    test_learning()