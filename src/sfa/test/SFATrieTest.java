package sfa.test;

import java.io.File;
import java.io.IOException;
import java.util.List;

import sfa.index.SFATrie;
import sfa.index.SortedListMap;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

public class SFATrieTest {

  public static void testSFATrie() throws IOException {
    int l = 16; // SFA word length ( & dimensionality of the index)
    int windowLength = 256; // length of the subsequences to be indexed
    int leafThreshold = 100; // number of subsequences in each leaf node
    int k = 1; // k-NN search

    System.out.println("Loading Time Series");
    TimeSeries timeSeries = TimeSeriesLoader.readSamplesSubsequences(new File("./datasets/indexing/labeled_lightcurves1.txt"),1);
    System.out.println("Sample DS size : " + timeSeries.getLength());

    TimeSeries timeSeries2 = TimeSeriesLoader.readSamplesSubsequences(new File("./datasets/indexing/burst.dat"),0);
    System.out.println("Query DS size : " + timeSeries2.getLength());

    SFATrie index = new SFATrie(l, leafThreshold);
    index.buildIndex(timeSeries, windowLength);
    index.checkIndex();

    // GC
    try {
      System.gc();
      Thread.sleep(10);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }

    System.out.println("Size of the index: "+index.getSize());

    int size = (timeSeries.getData().length-windowLength)+1;
    double[] means = new double[size];
    double[] stds = new double[size];
    TimeSeries.calcIncreamentalMeanStddev(windowLength, timeSeries, means, stds);

    for (int i = 0; i < 10; i++) {
      System.out.println(i+". Query");

      TimeSeries query = timeSeries2.getSubsequence(i, windowLength);

      long time = System.currentTimeMillis();
      SortedListMap<Double, Integer> result = index.searchNearestNeighbor(query, k);
      time = System.currentTimeMillis() - time;
      System.out.println("\tSFATree:" + (time/1000.0) + "s");

//      List<Integer> offsets = result.values();
      List<Double> distances = result.keys();

//      for (int j = 0; j < offsets.size(); j++) {
//        System.out.println("\tResult:\t"+distances.get(j) + "\t" +offsets.get(j));
//      }

      System.out.println("\tTS seen: " +index.getTimeSeriesRead());
      System.out.println("\tLeaves seen " + index.getIoTimeSeriesRead());
      System.out.println("\tNodes seen " +  index.getBlockRead());

      index.resetIoCosts();

//      if (!result.contains(query)) {
//        System.out.println("Error: ts not found!<");
//      }

      // compare with nearest neighbor search!
      time = System.currentTimeMillis();
      double resultDistance = Double.MAX_VALUE;
      for (int ww = 0; ww < size; ww++) { // faster than reevaluation in for loop
        double distance = getEuclideanDistance(timeSeries, query, means[ww], stds[ww], resultDistance, ww);
        resultDistance = Math.min(distance, resultDistance);
      }
      time = System.currentTimeMillis() - time;
      System.out.println("\tEuclidean:" + (time/1000.0) + "s");

      if (distances.get(0) != resultDistance) {
        System.out.println("\tError! Distances do not match: " + resultDistance + "\t" + distances.get(0));
      }
      else {
        System.out.println("\tDistance is ok");
      }
    }

    System.out.println("All ok...");
  }

  public static double getEuclideanDistance(
      TimeSeries ts,
      TimeSeries q,
      double meanTs,
      double stdTs,
      double minValue,
      int w
      ) {

    // 1 divided by stddev for fastert calculations
    stdTs = (stdTs>0? 1.0 / stdTs : 1.0);

    double distance = 0.0;
    double[] tsData = ts.getData();
    double[] qData = q.getData();

    for (int ww = 0; ww < qData.length; ww++) {
      double value1 = (tsData[w+ww]-meanTs) * stdTs;
      double value = qData[ww] - value1;
      distance += value*value;

      // early abandoning
      if (distance >= minValue) {
        return Double.MAX_VALUE;
      }
    }

    return distance;
  }

  public static void main(String argv[]) throws IOException {
    testSFATrie();
  }
}
