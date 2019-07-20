package sfa.index;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import sfa.SFAWordsTest;
import sfa.index.SFATrie;
import sfa.index.SortedListMap;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

/**
 * Performs a 1-NN search using the SFA trie
 */
@RunWith(JUnit4.class)
public class SFATrieTest {
  final static int l = 20; // SFA word queryLength ( & dimensionality of the index)
  final static int leafThreshold = 10; // number of subsequences in each leaf node
  final static int k = 1; // k-NN search

  public static void testWholeMatching() throws IOException {
    int N = 10_000;
    ClassLoader classLoader = SFAWordsTest.class.getClassLoader();
    TimeSeries[] timeSeries2 = TimeSeriesLoader
        .readSamplesQuerySeries(classLoader
            .getResource("datasets/indexing/query_lightcurves.txt")
            .getFile());
    int n = timeSeries2[0].getLength();
    System.out.println("Queries: " + timeSeries2.length);

    System.out.println("Generating Time Series");
    TimeSeries[] timeSeries = new TimeSeries[N];
    for (int i = 0; i < N; i++) {
      timeSeries[i] = TimeSeriesLoader.generateRandomWalkData(n, new Random(i));
      timeSeries[i].norm();
    }
    System.out.println("Whole Series: " + timeSeries.length);

    Runtime runtime = Runtime.getRuntime();
    long mem = runtime.totalMemory();

    SFATrie index = new SFATrie(l, leafThreshold);
    index.buildIndexWholeMatching(timeSeries);
    index.checkIndex();

    // GC
    performGC();
    System.out.println("Memory: " + ((runtime.totalMemory() - mem) / (1_048_576L)) + " MB (rough estimate)");

    System.out.println("Perform NN-queries");
    for (int i = 0; i < timeSeries2.length; i++) {
      System.out.println((i+1) + ". Query");
      TimeSeries query = timeSeries2[i];

      long time = System.currentTimeMillis();
      SortedListMap<Double, Integer> result = index.searchNearestNeighbor(query, k);
      time = System.currentTimeMillis() - time;
      System.out.println("\tSFATree:" + (time/1000.0) + "s");

      List<Double> distances = result.keys();

      //System.out.println("\tTS seen: " + index.getTimeSeriesRead() + " " + index.getTimeSeriesRead()/(double)N + "%");
      //System.out.println("\tLeaves seen " + index.getIoTimeSeriesRead());
      //System.out.println("\tNodes seen " +  index.getBlockRead());

      index.resetIoCosts();

      // compare with 1-NN ED search
      time = System.currentTimeMillis();
      double resultDistance = Double.MAX_VALUE;
      for (int w = 0; w < N; w++) {
        double distance = getEuclideanDistance(timeSeries[w], query, 0.0, 1.0, resultDistance, 0);
        resultDistance = Math.min(distance, resultDistance);
      }
      time = System.currentTimeMillis() - time;
      System.out.println("\tEuclidean:" + (time/1000.0) + "s");

      Assert.assertEquals("Distances do not match: " + resultDistance + "\t" + distances.get(0),
          distances.get(0), resultDistance, 0.003);

      //System.out.println("\tDistance is ok");
    }

    System.out.println("All ok...");
  }

  public static void testSubsequenceMatching() throws IOException {
    System.out.println("Loading Time Series");

    ClassLoader classLoader = SFAWordsTest.class.getClassLoader();
    TimeSeries[] timeSeries2 = TimeSeriesLoader.readSamplesQuerySeries(
        classLoader.getResource("datasets/indexing/query_lightcurves.txt").getFile());
    
    // TimeSeries timeSeries = TimeSeriesLoader.readSampleSubsequence(classLoader
    //                                    .getResource("datasets/indexing/sample_lightcurves.txt")
    //                                    .getFile());

    TimeSeries timeSeries = TimeSeriesLoader.generateRandomWalkData(100000, new Random(1));
    System.out.println("Sample DS size : " + timeSeries.getLength());

    int windowLength = timeSeries2[0].getLength(); // queryLength of the subsequences to be indexed
    System.out.println("Query DS size : " + windowLength);

    Runtime runtime = Runtime.getRuntime();
    long mem = runtime.totalMemory();
    long time = System.currentTimeMillis();

    SFATrie index = new SFATrie(l, leafThreshold);
    index.buildIndexSubsequenceMatching(timeSeries, windowLength);
    index.checkIndex();

    // GC
    performGC();
    System.out.println("Memory: " + ((runtime.totalMemory() - mem) / (1_048_576L)) + " MB (rough estimate)");

    System.out.println("Perform NN-queries");
    int size = (timeSeries.getData().length-windowLength)+1;
    double[] means = new double[size];
    double[] stds = new double[size];
    TimeSeries.calcIncrementalMeanStddev(windowLength, timeSeries.getData(), means, stds);

    for (int i = 0; i < timeSeries2.length; i++) {
      System.out.println((i+1) + ". Query");
      TimeSeries query = timeSeries2[i];

      time = System.currentTimeMillis();
      SortedListMap<Double, Integer> result = index.searchNearestNeighbor(query, k);
      time = System.currentTimeMillis() - time;
      System.out.println("\tSFATree:" + (time/1000.0) + "s");

      List<Double> distances = result.keys();

      //System.out.println("\tTS seen: " + index.getTimeSeriesRead() + " " +
      //    String.format("%.3f", index.getTimeSeriesRead()/(double)size) + "%");
      //System.out.println("\tLeaves seen " + index.getIoTimeSeriesRead());
      //System.out.println("\tNodes seen " +  index.getBlockRead());

      index.resetIoCosts();

      // compare with 1-NN ED search
      time = System.currentTimeMillis();
      double resultDistance = Double.MAX_VALUE;
      for (int ww = 0; ww < size; ww++) {
        double distance = getEuclideanDistance(timeSeries, query, means[ww], stds[ww], resultDistance, ww);
        resultDistance = Math.min(distance, resultDistance);
      }
      time = System.currentTimeMillis() - time;
      System.out.println("\tEuclidean:" + (time/1000.0) + "s");

      Assert.assertEquals("Distances do not match: " + resultDistance + "\t" + distances.get(0),
          distances.get(0), resultDistance, 0.003);
    }

    System.out.println("All ok...");
  }

  public static void testSubsequenceMatchingRangeQuery() throws IOException {
    System.out.println("Loading Time Series");

    ClassLoader classLoader = SFAWordsTest.class.getClassLoader();
    TimeSeries[] timeSeries2 = TimeSeriesLoader.readSamplesQuerySeries(
        classLoader.getResource("datasets/indexing/query_lightcurves.txt").getFile());

    TimeSeries timeSeries = TimeSeriesLoader.generateRandomWalkData(100000, new Random(1));
    System.out.println("Sample DS size : " + timeSeries.getLength());

    int windowLength = timeSeries2[0].getLength(); // queryLength of the subsequences to be indexed
    System.out.println("Query DS size : " + windowLength);

    Runtime runtime = Runtime.getRuntime();
    long mem = runtime.totalMemory();
    long time = System.currentTimeMillis();

    SFATrie index = new SFATrie(l, leafThreshold);
    index.buildIndexSubsequenceMatching(timeSeries, windowLength);
    index.checkIndex();

    // GC
    performGC();
    System.out.println("Memory: " + ((runtime.totalMemory() - mem) / (1_048_576L)) + " MB (rough estimate)");

    System.out.println("Perform NN-queries");
    int size = (timeSeries.getData().length-windowLength)+1;
    double[] means = new double[size];
    double[] stds = new double[size];
    TimeSeries.calcIncrementalMeanStddev(windowLength, timeSeries.getData(), means, stds);

    for (int i = 0; i < timeSeries2.length; i++) {
      System.out.println((i+1) + ". Query");
      TimeSeries query = timeSeries2[i];

      // do a brute force range query search to set epsilon
      double epsilon = Double.MAX_VALUE;
      for (int ww = 0; ww < size; ww++) {
        double distance = getEuclideanDistance(timeSeries, query, means[ww], stds[ww], epsilon, ww);
        if (distance < epsilon) {
          epsilon = 1.001*distance;
        }
      }

      // set epsilon to be 1.05x the minimal distance
      long timeED = System.currentTimeMillis();
      int count = 0;
      for (int ww = 0; ww < size; ww++) {
        double distance = getEuclideanDistance(timeSeries, query, means[ww], stds[ww], epsilon, ww);
        if (distance <= epsilon) {
          count++;
        }
      }
      timeED = System.currentTimeMillis() - timeED;

      time = System.currentTimeMillis();
      List<Integer> result = index.searchEpsilonRange(query, epsilon);
      time = System.currentTimeMillis() - time;
      System.out.println("\tSFATree:" + (time/1000.0) + "s");
      index.resetIoCosts();

      for (Integer a : result) {
        System.out.println(a);
      }

      System.out.println("\tEuclidean:" + (timeED/1000.0) + "s");

      Assert.assertEquals("Counts do not match: " + result.size() + "\t" + count,
          result.size(), count, 0.000);
    }

    System.out.println("All ok...");
  }

  public static void performGC() {
    try {
      System.gc();
      Thread.sleep(10);
    } catch (InterruptedException e) {
      Assert.fail(e.getMessage());
    }
  }

  public static double getEuclideanDistance(
      TimeSeries ts,
      TimeSeries q,
      double meanTs,
      double stdTs,
      double minValue,
      int w
      ) {

    // 1 divided by stddev for faster calculations
    stdTs = (stdTs>0? 1.0 / stdTs : 1.0);

    double distance = 0.0;
    double[] tsData = ts.getData();
    double[] qData = q.getData();

    for (int ww = 0; ww < qData.length; ww++) {
      double value1 = (tsData[w+ww]-meanTs) * stdTs;
      double value = qData[ww] - value1;
      distance += value*value;

      // early abandoning
      if (distance > minValue) {
        return Double.MAX_VALUE;
      }
    }

    return distance;
  }

  @Test
  public void testSFATrieTest() throws IOException {
    testWholeMatching();
    testSubsequenceMatching();
    testSubsequenceMatchingRangeQuery();
  }
}
