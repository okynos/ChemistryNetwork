package neuronalnetwork;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.time.LocalDate;
import java.time.LocalTime;
import java.util.Locale;
import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;

/**
 * MNIST database utilities.
 * 
 * @author Fernando Berzal (berzal@acm.org)
 */
public class MNISTDatabase 
{
	// MNIST URL
	private static final String MNIST_URL = "http://yann.lecun.com/exdb/mnist/";
	
	// Training data
	private static final String trainingImages = "train-images-idx3-ubyte.gz";
	private static final String trainingLabels = "train-labels-idx1-ubyte.gz";
	
	// Test data
	private static final String testImages = "t10k-images-idx3-ubyte.gz";
	private static final String testLabels = "t10k-labels-idx1-ubyte.gz";
	
	// Logger
	protected static final Logger log = Logger.getLogger(MNISTDatabase.class.getName());
	
	// Download files
	
	/**
	 * Download URL to file using Java NIO.
	 * 
	 * @param url Source URL
	 * @param filename Destination file name
	 */
	public static void download (String urlString, String filename)
		throws IOException
	{
		URL url = new URL(urlString);
		ReadableByteChannel rbc = Channels.newChannel(url.openStream());
		FileOutputStream fos = new FileOutputStream(filename);
		fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
		fos.close();
		rbc.close();
	}

	/**
	 * Download MNIST database.
	 * 
	 * @param directory Destination folder/directory.
	 * @throws IOException
	 */
	public static void downloadMNIST (String directory) throws IOException 
	{
		File baseDir = new File(directory);
		
		if (!(baseDir.isDirectory() || baseDir.mkdir())) {
			throw new IOException("Unable to create destination folder " + baseDir);
		}
		
		log.info("Downloading MNIST database...");
		
		download(MNIST_URL+trainingImages, directory+trainingImages);
		download(MNIST_URL+trainingLabels, directory+trainingLabels);
		download(MNIST_URL+testImages, directory+testImages);
		download(MNIST_URL+testLabels, directory+testLabels);

		log.info("MNIST database downloaded into "+directory);
	}

	
	// Read data from files
	
	/**
	 * Read MNIST image data.
	 * 
	 * @param filename File name
	 * @return 3D int array
	 * @throws IOException
	 */
	public static int[][][] readImages (String filename)
			throws IOException
	{
		FileInputStream file = null;
		InputStream gzip = null;
		DataInputStream data = null;
		int images[][][] = null;

		try {
			file = new FileInputStream(filename);
			gzip = new GZIPInputStream(file);
			data = new DataInputStream(gzip);

			log.info("Reading MNIST data...");

			int magicNumber = data.readInt();

			if (magicNumber!=2051) // 0x00000801 == 08 (unsigned byte) + 03 (3D tensor, i.e. multiple 2D images)
				throw new IOException("Error while reading MNIST data from "+filename);

			int size = data.readInt();
			int rows = data.readInt();
			int columns = data.readInt();

			images = new int[size][rows][columns];
			
			log.info("Reading "+size+" "+rows+"x"+columns+" images...");

			for (int i=0; i<size; i++)
				for (int j=0; j<rows; j++)
					for (int k=0; k<columns; k++)
						images[i][j][k] = data.readUnsignedByte();

			log.info("MNIST images read from "+filename);

		} finally {

			if (data!=null)
				data.close();
			if (gzip!=null)
				gzip.close();
			if (file!=null)
				file.close();
		}

		return images;
	}
	
	/**
	 * Read MNIST labels
	 * 
	 * @param filename File name
	 * @return Label array
	 * @throws IOException
	 */
	public static int[] readLabels (String filename)
		throws IOException
	{
		FileInputStream file = null;
		InputStream gzip = null;
		DataInputStream data = null;
		int labels[] = null;

		try {
			file = new FileInputStream(filename);
			gzip = new GZIPInputStream(file);
			data = new DataInputStream(gzip);

			log.info("Reading MNIST labels...");

			int magicNumber = data.readInt();

			if (magicNumber!=2049) // 0x00000801 == 08 (unsigned byte) + 01 (vector)
				throw new IOException("Error while reading MNIST labels from "+filename);

			int size = data.readInt();
			
			labels = new int[size];

			for (int i=0; i<size; i++)
				labels[i] = data.readUnsignedByte();

			log.info("MNIST labels read from "+filename);

		} finally {
			
			if (data!=null)
				data.close();
			if (gzip!=null)
				gzip.close();
			if (file!=null)
				file.close();
		}
		
		return labels;
	}

	
	/**
	 * Normalize raw image data, i.e. convert to floating-point and rescale to [0,1].
	 * 
	 * @param image Raw image data
	 * @return Floating-point 2D array
	 */
	public static float[][] normalize (int image[][])
	{
		int rows = image.length;
		int columns = image[0].length;
		float data[][] = new float[rows][columns];
		
		for (int i=0; i<rows; i++)
			for (int j=0; j<rows; j++)
				data[i][j] = (float)image[i][j] / 255f;
		
		return data;
	}

	
	// Standard I/O
	
	public static String toString (int label)
	{
		return Integer.toString(label);
	}
	
	public static String toString (int image[][])
	{
		StringBuilder builder = new StringBuilder();

		for (int i=0; i<image.length; i++) {
			for (int j=0; j<image[i].length; j++) {
				String hex = Integer.toHexString(image[i][j]);
				if (hex.length()==1) 
					builder.append("0");
				builder.append(hex);
				builder.append(' ');				
			}
			builder.append('\n');
		}
		
		return builder.toString();
	}

	public static String toString (float image[][])
	{
		StringBuilder builder = new StringBuilder();

		for (int i=0; i<image.length; i++) {
			for (int j=0; j<image[i].length; j++) {
				builder.append( String.format(Locale.US, "%.3f ", image[i][j]) );
			}
			builder.append('\n');
		}
		
		return builder.toString();
	}

	// Test program
	
	public static void main (String[] args) throws IOException{
		boolean loadfile = false;
		String version = "v5.0";
		int IMAGESIZE = 28;
		int images[][][];
		int tImages[][][];
		images = readImages("data/mnist/"+trainingImages);
		tImages = readImages("data/mnist/"+testImages);
		
		
		int labels[], tLabels[];
		labels = readLabels("data/mnist/"+trainingLabels);
		tLabels = readLabels("data/mnist/"+testLabels);

		
		float trainData[][][] = new float[images.length][IMAGESIZE][IMAGESIZE];
		float testData[][][] = new float[tImages.length][IMAGESIZE][IMAGESIZE];
		// Normalize image data
		for(int i=0; i<trainData.length; ++i){
			 trainData[i] = normalize(images[i]);
		}
		for(int i=0; i<testData.length; ++i){
			testData[i] = normalize(tImages[i]);
		}
	
		
		NeuralNetwork net = new NeuralNetwork(!loadfile);
		if(!loadfile){
			System.out.println("Entrenando la red " + version);
			long startTime = System.currentTimeMillis();
			net.trainNetwork(trainData, labels, testData, tLabels);
			long stopTime = System.currentTimeMillis();
			System.out.println("Tiempo transcurrido de entrenamiento: " + (stopTime-startTime)/1000L + "s");
			
			System.out.println("Probando el conjunto de entrenamiento");
			net.testNetwork(trainData, labels);
			
			System.out.println("Probando un conjunto no entrenado");
			net.testNetwork(testData, tLabels);
			
			String filename = Integer.toString(LocalDate.now().getDayOfMonth()) + "_" 
			+ Integer.toString(LocalTime.now().getHour()) + "-" 
					+ Integer.toString(LocalTime.now().getMinute()) + ".txt";
			System.out.println("Guardando Pesos en el fichero '" + filename + "'");
			net.saveWeights(filename);
		}else{
			net.loadWeights("7_20-38.txt");
			net.testNetwork(trainData, labels);
			net.testNetwork(testData, tLabels);
			System.out.println(net.getLabels());
		}
		
	}

}
