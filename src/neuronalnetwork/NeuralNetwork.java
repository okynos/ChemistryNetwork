package neuronalnetwork;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.logging.Logger;

public class NeuralNetwork {
	protected static final Logger log = Logger.getLogger(NeuralNetwork.class.getName());
	private final int IMAGESIZE = 28;
	private final int epochs = 60;
	private final float learningRate = 0.017f;
	private final float momentum = 0.9f;
	private final float maxRandom = 0.1f;
	private final float minRandom = -0.1f;
	private int numInputNeurons = 64;
	private int numHiddenNeurons = 32;
	private final int numOutputNeurons = 10;
	private int hits = 0;
	private String resultLabels;
	
	
	private float[][][] inputWeightArray;
	private float[][][] inputInpArray;
	private float[] inputOutArray;
	private float[] inputOutError;
	private float[] inputInpError;
	private float[][][] inputWeightError;
	
	private float[][] hiddenWeightArray;
	private float[][] hiddenInpArray;
	private float[] hiddenOutArray;
	private float[] hiddenOutError;
	private float[] hiddenInpError;
	private float[][] hiddenWeightError;
	
	private float[][] outputWeightArray;
	private float[][] outputInpArray;
	private float[] outputOutArray;
	private float[] outputOutError;
	private float[] outputInpError;
	private float[][] outputWeightError;
	
	public NeuralNetwork(boolean initialize){
		if(initialize)
			initializeArrays();
	}
	
	private void initializeArrays(){
		inputWeightArray = new float[numInputNeurons][IMAGESIZE][IMAGESIZE];
		inputInpArray = new float[numInputNeurons][IMAGESIZE][IMAGESIZE];
		inputOutArray = new float[numInputNeurons];
		inputOutError = new float[numInputNeurons];
		inputInpError = new float[numInputNeurons];
		inputWeightError = new float[numInputNeurons][IMAGESIZE][IMAGESIZE];
		
		hiddenWeightArray = new float [numHiddenNeurons][numInputNeurons];
		hiddenInpArray = new float[numHiddenNeurons][numInputNeurons];
		hiddenOutArray = new float[numHiddenNeurons];
		hiddenOutError = new float[numHiddenNeurons];
		hiddenInpError = new float[numHiddenNeurons];
		hiddenWeightError = new float[numHiddenNeurons][numInputNeurons];
		
		outputWeightArray = new float [numOutputNeurons][numHiddenNeurons];
		outputInpArray = new float[numOutputNeurons][numHiddenNeurons];
		outputOutArray = new float[numOutputNeurons];
		outputOutError = new float[numOutputNeurons];
		outputInpError = new float[numOutputNeurons];
		outputWeightError = new float[numOutputNeurons][numHiddenNeurons];
		
		Random r = new Random();
		for(int i=0; i<numInputNeurons; ++i){
			for(int j=0; j<IMAGESIZE; ++j){
				for(int k=0; k<IMAGESIZE; ++k){
					inputWeightArray[i][j][k] = minRandom + (maxRandom - minRandom) * r.nextFloat();
					inputWeightError[i][j][k] = 0.0f;
				}
			}
		}
		
		for(int i=0; i<numHiddenNeurons; ++i){
			for(int j=0; j<numInputNeurons; ++j){
				hiddenWeightArray[i][j] = minRandom + (maxRandom - minRandom) * r.nextFloat();
				hiddenWeightError[i][j] = 0.0f;
			}
		}
		
		for(int i=0; i<numOutputNeurons; ++i){
			for(int j=0; j<numHiddenNeurons; ++j){
				outputWeightArray[i][j] = minRandom + (maxRandom - minRandom) * r.nextFloat();
				outputWeightError[i][j] = 0.0f;
			}
		}
	}
	
	private float sigmoid(float value){
		return (float) (1.0 / (1.0 + Math.exp(-value)));
	}
	
	/*private float noise(float b, float sigma, float value){
		return (float) ((1 / (sigma*Math.sqrt(2*Math.PI))) * Math.exp(-0.5f * (value-b)*(value-b) / (sigma*sigma) ) );
	}*/
	
	public void testNetwork(float[][][] data, int[] label){
		hits = 0;
		resultLabels = "";
		for(int i=0; i<data.length; ++i){
			testImage(data[i], label[i]);
		}
		float tasaError = (1f-(float)hits/data.length);
		System.out.println(" Aciertos: " + hits + ", Tasa de error: " + tasaError*100 + "%");
	}
	
	private void testImage(float[][] image, int label){
		insertImage(image);
		sumInputAndWeightsINP();
		
		forwardPropagationMiddle();
		sumInputAndWeightsHID();
		forwardPropagationOutside();
		sumInputAndWeightsOUT();
		
		int output = networkOutput();
		checkLabel(output, label);
		resultLabels += Integer.toString(output);
	}
	
	public void trainNetwork(float[][][] data, int[] label, float[][][] testData, int[] testLabel){
		int output;
		float tasaError;
		for(int e=0; e<epochs; ++e){
			hits = 0;
			for(int i=0; i<data.length; ++i){
				trainImage(data[i], label[i]);
				output = networkOutput();
				checkLabel(output, label[i]);
			}
			tasaError = (1f-(float)hits/data.length);
			System.out.println("Época " + e + " -> Aciertos: " + hits + ", Tasa de error: " + tasaError*100 + "%");
			testNetwork(testData, testLabel);
		}
	}
	
	
	private void trainImage(float[][] image, int label){
		insertImage(image);
		sumInputAndWeightsINP();
		
		forwardPropagationMiddle();
		sumInputAndWeightsHID();
		forwardPropagationOutside();
		sumInputAndWeightsOUT();
		
		backpropagate(label);
	}
	
	private void insertImage(float[][] image){
		for(int i=0; i<numInputNeurons; ++i){
			inputInpArray[i] = image;
		}
	}
	
	private int networkOutput(){
		float maximo = 0.0f;
		int output = 0;
		for(int s=0; s<numOutputNeurons; ++s){
			if(outputOutArray[s] > maximo){
				maximo = outputOutArray[s];
				output = s;
			}
		}
		return output;
	}
	
	private void checkLabel(int output, int label){
		if(output == label)
			hits++;
	}
	
	private void sumInputAndWeightsINP(){
		float sum = 0f;
		for(int i=0; i<numInputNeurons; ++i){
			sum = 0f;
			for(int j=0; j<IMAGESIZE; ++j){
				for(int k=0; k<IMAGESIZE; ++k){
					sum += inputWeightArray[i][j][k] * (inputInpArray[i][j][k] 
							/*+ noise(0.1f, 0.6f, inputInpArray[i][j][k])*/);
				}
			}
			inputOutArray[i] = sigmoid(sum);
		}
	}
	
	private void sumInputAndWeightsHID(){
		float sum = 0f;
		for(int i=0; i<numHiddenNeurons; ++i){
			sum = 0f;
			for(int j=0; j<numInputNeurons; ++j){
				sum += hiddenWeightArray[i][j] * hiddenInpArray[i][j];
			}
			hiddenOutArray[i] = sigmoid(sum);
		}
	}
	
	private void sumInputAndWeightsOUT(){
		float sum = 0f;
		for(int i=0; i<numOutputNeurons; ++i){
			sum = 0f;
			for(int j=0; j<numHiddenNeurons; ++j){
				sum += outputWeightArray[i][j] * outputInpArray[i][j];
			}
			outputOutArray[i] = sigmoid(sum);
		}
	}
	
	private void forwardPropagationMiddle(){
		for(int i=0; i<numHiddenNeurons; ++i){
			for(int j=0; j<inputOutArray.length; ++j){
				hiddenInpArray[i][j] = inputOutArray[j];
			}
		}
	}
	
	private void forwardPropagationOutside(){
		for(int i=0; i<numOutputNeurons; ++i){
			for(int j=0; j<hiddenOutArray.length; ++j){
				outputInpArray[i][j] = hiddenOutArray[j];
			}
		}
	}
	
	private void backpropagate(int label){
		backpropagateOUT(label);
		
		backpropagateHID();
		
		backpropagateINP();
		adjustWeightINP();
		
		adjustWeightHID();
		
		adjustWeightOUT();
	}
	
	private void backpropagateOUT(int label){
		calculateOutputErrorOUT(label);
		calculateInputErrorOUT();
		calculateWeightErrorOUT();
	}
	
	private void backpropagateHID(){
		calculateOutputErrorHID();
		calculateInputErrorHID();
		calculateWeightErrorHID();
	}
	
	private void backpropagateINP(){
		calculateOutputErrorINP();
		calculateInputErrorINP();
		calculateWeightErrorINP();
	}

	
	
	
	/*
	 * Métodos de cálculo de error de salida
	 */	
	private void calculateOutputErrorOUT(int label){
		for(int i=0; i<numOutputNeurons; ++i){
			if(label == i){
				outputOutError[i] = 1.0f - outputOutArray[i];
			}else{
				outputOutError[i] = 0.0f - outputOutArray[i];
			}
		}
	}
	
	private void calculateInputErrorOUT(){
		for(int i=0; i<numOutputNeurons; ++i){
			outputInpError[i] = outputOutArray[i] * (1.0f - outputOutArray[i]) * outputOutError[i];
		}
	}
	
	private void calculateWeightErrorOUT(){
		for(int i=0; i<numOutputNeurons; ++i){
			for(int j=0; j<numHiddenNeurons; ++j){
				outputWeightError[i][j] = (outputInpError[i] * hiddenOutArray[j] * learningRate) 
						+ momentum * outputWeightError[i][j];
			}
		}
	}
	
	private void adjustWeightOUT(){
		float weightSum;
		for(int i=0; i<numOutputNeurons; ++i){
			for(int j=0; j<numHiddenNeurons; ++j){
				weightSum = outputWeightArray[i][j] + outputWeightError[i][j];
				if(weightSum < 1f && weightSum > -1f){
					outputWeightArray[i][j] = weightSum;
				}
			}
		}
	}
	
	
	
	
	
	
	
	
	/*
	 * Métodos de cálculo de error en la capa intermedia
	 */
	private void calculateOutputErrorHID(){
		for(int i=0; i<numHiddenNeurons; ++i){
			hiddenOutError[i] = 0f;
			for(int j=0; j<numOutputNeurons; ++j){
				hiddenOutError[i] = hiddenOutError[i] + outputInpError[j] * outputWeightArray[j][i];
			}
		}
	}
	
	private void calculateInputErrorHID(){
		for(int i=0; i<numHiddenNeurons; ++i){
			hiddenInpError[i] = hiddenOutArray[i] * (1.0f - hiddenOutArray[i]) * hiddenOutError[i];
		}
	}
	
	private void calculateWeightErrorHID(){
		for(int i=0; i<numHiddenNeurons; ++i){
			for(int j=0; j<numInputNeurons; ++j){
				hiddenWeightError[i][j] = (hiddenInpError[i] * inputOutArray[j] * learningRate) 
						+ momentum * hiddenWeightError[i][j];
			}
		}
	}
	
	private void adjustWeightHID(){
		float weightSum;
		for(int i=0; i<numHiddenNeurons; ++i){
			for(int j=0; j<numInputNeurons; ++j){
				weightSum = hiddenWeightArray[i][j] + hiddenWeightError[i][j];
				if(weightSum < 1f && weightSum > -1f){
					hiddenWeightArray[i][j] = weightSum;
				}
			}
		}
		
	}
	
	
	
	
	
	
	
	
	
	/*
	 * Métodos de cálculo de error en la capa intermedia de entrada
	 */
	private void calculateOutputErrorINP(){
		for(int i=0; i<numInputNeurons; ++i){
			inputOutError[i] = 0f;
			for(int j=0; j<numHiddenNeurons; ++j){
				inputOutError[i] = inputOutError[i] + hiddenInpError[j] * hiddenWeightArray[j][i];
			}
		}
	}
	
	private void calculateInputErrorINP(){
		for(int i=0; i<numInputNeurons; ++i){
			inputInpError[i] = inputOutArray[i] * (1.0f - inputOutArray[i]) * inputOutError[i];
		}
	}
	
	private void calculateWeightErrorINP(){
		for(int i=0; i<numInputNeurons; ++i){
			for(int j=0; j<IMAGESIZE; ++j){
				for(int k=0; k<IMAGESIZE; ++k){
					inputWeightError[i][j][k] = (inputInpError[i] * inputInpArray[i][j][k] * learningRate ) 
							+ momentum * inputWeightError[i][j][k];
				}
			}
		}
	}
	
	private void adjustWeightINP(){
		float weightSum;
		for(int i=0; i<numInputNeurons; ++i){
			for(int j=0; j<IMAGESIZE; ++j){
				for(int k=0; k<IMAGESIZE; ++k){
					weightSum = inputWeightArray[i][j][k] + inputWeightError[i][j][k];
					if(weightSum < 1f && weightSum > -1f){
						inputWeightArray[i][j][k] += inputWeightError[i][j][k];
					}
				}
			}
		}
		
	}
	
	
	
	
	/*
	 * Métodos para cargar o guardar los pesos.
	 */
	
	public void saveWeights(String filename) throws IOException{
		  BufferedWriter outputWriter = new BufferedWriter(new FileWriter(filename));
		  outputWriter.write(Integer.toString(numInputNeurons));
		  outputWriter.newLine();
		  outputWriter.write(Integer.toString(numHiddenNeurons));
		  outputWriter.newLine();
		  for (int i = 0; i < inputWeightArray.length; i++) {
			  for(int j=0; j< inputWeightArray[i].length; ++j){
				  for(int k=0; k<inputWeightArray[i][j].length; ++k){
					    outputWriter.write(Float.toString(inputWeightArray[i][j][k]));
					    outputWriter.newLine();
				  }
			  }
		  }
		  for (int i = 0; i < hiddenWeightArray.length; i++) {
			  for(int j=0; j< hiddenWeightArray[i].length; ++j){
				  outputWriter.write(Float.toString(hiddenWeightArray[i][j]));
				  outputWriter.newLine();
			  }
		  }
		  for (int i = 0; i < outputWeightArray.length; i++) {
			  for(int j=0; j< outputWeightArray[i].length; ++j){
				  outputWriter.write(Float.toString(outputWeightArray[i][j]));
				  outputWriter.newLine();
			  }
		  }
		  outputWriter.flush();  
		  outputWriter.close(); 
		  log.info("Data Saved in " + filename);
	}
	
	public void loadWeights(String filename) throws IOException{
		  BufferedReader inputReader = new BufferedReader(new FileReader(filename));
		  numInputNeurons = Integer.parseInt(inputReader.readLine());
		  numHiddenNeurons = Integer.parseInt(inputReader.readLine());
		  initializeArrays();
		  for (int i = 0; i < inputWeightArray.length; i++) {
			  for(int j=0; j< inputWeightArray[i].length; ++j){
				  for(int k=0; k<inputWeightArray[i][j].length; ++k){
					  inputWeightArray[i][j][k] = Float.parseFloat(inputReader.readLine());// .write(Float.toString(inputWeightArray[i][j][k]));
				  }
			  }
		  }
		  for (int i = 0; i < hiddenWeightArray.length; i++) {
			  for(int j=0; j< hiddenWeightArray[i].length; ++j){
				  hiddenWeightArray[i][j] = Float.parseFloat(inputReader.readLine());//outputWriter.write(Float.toString(outputWeightArray[i][j]));
			  }
		  } 
		  for (int i = 0; i < outputWeightArray.length; i++) {
			  for(int j=0; j< outputWeightArray[i].length; ++j){
				  outputWeightArray[i][j] = Float.parseFloat(inputReader.readLine());//outputWriter.write(Float.toString(outputWeightArray[i][j]));
			  }
		  }  
		  inputReader.close(); 
		  log.info("Data Readed");
	}
	
	public String getLabels(){
		return resultLabels;
	}
	
	
}
