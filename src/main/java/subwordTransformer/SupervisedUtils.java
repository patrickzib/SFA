package subwordTransformer;

public class SupervisedUtils {
  public static double sigma(double[] supports, double maxSigma) {
    int numClasses = supports.length;
    double supportSum = 0;
    for (int i = 0; i < numClasses; i++) {
      supportSum += supports[i];
    }
    double[] confidences = new double[numClasses];
    double confidenceAvg = 0;
    for (int i = 0; i < numClasses; i++) {
      double confidence = supports[i] / supportSum;
      confidences[i] = confidence;
      confidenceAvg += confidence;
    }
    confidenceAvg /= numClasses;
    double sigma = 0;
    for (int i = 0; i < numClasses; i++) {
      sigma += Math.pow(confidences[i] - confidenceAvg, 2);
    }
    sigma /= numClasses;
    return Math.sqrt(sigma) / maxSigma;
  }

  public static double sigma(int[] wordCounts, int[] classSizes, double maxSigma) {
    int numClasses = wordCounts.length;
    double[] supports = new double[numClasses];
    for (int i = 0; i < numClasses; i++) {
      supports[i] = ((double) wordCounts[i]) / classSizes[i];
    }
    return sigma(supports, maxSigma);
  }
}
