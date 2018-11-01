// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sfa.index.SFATrie;
import sfa.index.SFATrieTest;
import sfa.index.SortedListMap;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;
import sfa.transformation.SFA;
import sfa.transformation.SFA.HistogramType;

import java.io.*;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;

@RunWith(JUnit4.class)
public class SFABulkLoadTest {

  static File tempDir = null;
  static ExecutorService serializerExec = Executors.newFixedThreadPool(2); // serialize access to the disk
  static ExecutorService transformExec = Executors.newFixedThreadPool(4); // parallel SFA transformation

  static LinkedList<Future<Long>> futures = new LinkedList<>();

  static int l = 16; // SFA word queryLength ( & dimensionality of the index)
  static int leafThreshold = 100; // number of subsequences in each leaf node
  static byte symbols = SFATrie.symbols;

  static Runtime runtime = Runtime.getRuntime();

  @Before
  public void setUpBucketDir() {
    try {
      tempDir = Files.createTempDirectory("tmp").toFile();
      System.out.println("Created temp directory at "+tempDir.getAbsolutePath());

      // remove file on exit
      tempDir.deleteOnExit();
    } catch (IOException e) {
      Assert.fail("Unable to create temp directory: " + e.getMessage());
    }
  }

  @Test
  public void testBulkLoadWholeMatching() throws IOException {

    // int N = 100000;
    // System.out.println("Loading/generating "+N+" Time Series...");
    //
    // TimeSeries[] timeSeries2 = TimeSeriesLoader.readSamplesQuerySeries(
    // new File("./datasets/indexing/query_lightcurves.txt"));
    // int n = timeSeries2[0].getLength();
    // System.out.println("Queries DS size: " + timeSeries2.queryLength);
    //
    // long mem = runtime.totalMemory();
    //
    // // train SFA quantization bins on a SUBSET of all time series,
    // // as we cannot fitEnsemble all into main memory
    // TimeSeries[] timeSeriesSubset = new TimeSeries[50_000];
    // for (int i = 0; i < N; i++) {
    // timeSeriesSubset[i] = getTimeSeries(i, n);
    // timeSeriesSubset[i].norm();
    // }
    // System.out.println("Sample DS size: " + N);
    //
    // SFA sfa = new SFA(HistogramType.EQUI_FREQUENCY);
    // sfa.fitTransform(timeSeriesSubset, l, symbols, true);
    // // sfa.printBins();
    //
    // // process data in chunks of 'chunkSize' to create one SFA trie each
    // int chunkSize = 10_000;
    // System.out.println("Chunk size:\t" + chunkSize);
    // int trieDepth = getBestDepth(N, chunkSize);
    //
    // // write the Fourier-transformed TS to buckets on disk
    // SerializedStreams dataStream = new SerializedStreams(trieDepth);
    //
    // // process in parallel, as the Fourier transform of whole series takes
    // O(n log n)
    // int BLOCKS = (int)Math.ceil(N/chunkSize);
    // ParallelFor.withIndex(transformExec, BLOCKS, new ParallelFor.Each() {
    // long time = System.currentTimeMillis();
    // @Override
    // public void run(int id, AtomicInteger processed) {
    // for (int i = 0, a = 0; i < N; i+=chunkSize, a++) { // process TS in
    // chunkSize
    // if (a % BLOCKS == id) {
    // System.out.println("Transforming Chunk: " + (a + 1));
    // for (int j = i; j < Math.min(i+chunkSize, N); j++) {
    // // could as well load TS from file
    // TimeSeries ts = getTimeSeries(j, n);
    // // Fourier transform
    // double[] words = sfa.transformation.transform(ts, l);
    // // SFA words
    // byte[] w = sfa.quantizationByte(words);
    // dataStream.addToPartition(w, words, j, trieDepth);
    // }
    //
    // // wait for all futures to finish (data to be written)
    // long bytesWritten = 0;
    // while (!futures.isEmpty()) {
    // try {
    // bytesWritten = futures.remove().get();
    // } catch (Exception e) {
    // e.printStackTrace();
    // }
    // }
    // System.out.println("\tavg write speed: " + (bytesWritten /
    // (System.currentTimeMillis() - time)) + " kb/s");
    // }
    // }
    //
    // }
    // });
    // // close all streams
    // dataStream.setFinished();
    //
    // // build the SFA trie from the bucket files
    // SFATrie index = buildSFATrie(l, leafThreshold, n, trieDepth, sfa);
    //
    // // add the raw data to the trie
    // // TODO ?? index.initializeWholeMatching(timeSeries);
    // index.printStats();
    //
    // // GC
    // performGC();
    // System.out.println("Memory: " + ((runtime.totalMemory() - mem) /
    // (1048576l)) + " MB (rough estimate)");

  }

  /**
   * Gets the i-th time series of queryLength n
   *
   * @param i
   * @param n
   * @return
   */
  private TimeSeries getTimeSeries(int i, int n) {
    return TimeSeriesLoader.generateRandomWalkData(n, new Random(i));
  }

  /**
   * the depth of the tree to use for bulk loading: depth 1 => 8^1 buckets depth
   * 2 => 8^2 buckets ... depth i => symbols ^ i buckets
   *
   * @param count
   * @param chunkSize
   * @return
   */
  private int getBestDepth(int count, int chunkSize) {
    int trieDepth = (int) (Math.round(Math.log(count / chunkSize) / Math.log(8)));
    System.out.println("Using trie depth:\t" + trieDepth + " (" + (int) Math.pow(8, trieDepth) + " buckets)");
    return trieDepth;
  }

  @Test
  public void testBulkLoadSubsequenceMatching() throws IOException {
    int N = 20 * 100_000;
    System.out.println("Loading/generating Time Series of queryLength " + N + "...");

    // samples to be indexed
    TimeSeries timeSeries = getTimeSeries(1, N);
    System.out.println("Sample DS size:\t" + N);

    // query subsequences
    ClassLoader classLoader = SFAWordsTest.class.getClassLoader();

    TimeSeries[] timeSeries2 = TimeSeriesLoader.readSamplesQuerySeries(
        classLoader.getResource("datasets/indexing/query_lightcurves.txt").getFile());
    int n = timeSeries2[0].getLength();
    System.out.println("Query DS size:\t" + n);

    long mem = runtime.totalMemory();

    // train SFA quantization bins on the whole time series
    SFA sfa = new SFA(HistogramType.EQUI_FREQUENCY);
    sfa.fitWindowing(new TimeSeries[]{timeSeries}, n, l, symbols, true, true);
    // sfa.printBins();

    // process data in chunks of 'chunkSize' and create one index each
    int chunkSize = 100_000;
    System.out.println("Chunk size:\t" + chunkSize);
    int trieDepth = getBestDepth(N, chunkSize);

    // write the Fourier-transformed TS to buckets on disk
    SerializedStreams dataStream = new SerializedStreams(trieDepth);
    long time = System.currentTimeMillis();

    // transform all approximations
    // no need to transform in parallel, as the Momentary Fourier transform
    // transforms
    // each subsequence in constant time
    for (int i = 0, a = 0; i < timeSeries.getLength(); i += chunkSize, a++) {
      //System.out.println("Transforming Chunk: " + (a + 1));
      TimeSeries subsequence = timeSeries.getSubsequence(i, chunkSize + n - 1);
      double[][] words = sfa.transformWindowingDouble(subsequence);
      for (int pos = 0; pos < words.length; pos++) {
        byte[] w = sfa.quantizationByte(words[pos]);
        dataStream.addToPartition(w, words[pos], i + pos, trieDepth);
      }

      // wait for all tasks to finish
      long bytesWritten = 0;
      while (!futures.isEmpty()) {
        try {
          bytesWritten = futures.remove().get();
        } catch (Exception e) {
          Assert.fail(e.getMessage());
        }
      }
      System.out.println("\tavg write speed: " + (bytesWritten / (System.currentTimeMillis() - time)) + " kb/s");
    }

    // close all streams
    dataStream.setFinished();

    // build the SFA trie from the bucket files
    SFATrie index = buildSFATrie(l, leafThreshold, n, trieDepth, sfa);

    // add the raw data to the trie
    index.initializeSubsequenceMatching(timeSeries, n);
    //index.printStats();

    // GC
    performGC();
    System.out.println("Memory: " + ((runtime.totalMemory() - mem) / (1_048_576L)) + " MB (rough estimate)");

    // k-NN search
    int k = 1;

    // Used for Euclidean distance computations
    int size = (timeSeries.getData().length - n) + 1;
    double[] means = index.means;
    double[] stds = index.stddev;

    for (int i = 0; i < timeSeries2.length; i++) {
      System.out.println((i + 1) + ". Query");
      TimeSeries query = timeSeries2[i];

      time = System.currentTimeMillis();
      SortedListMap<Double, Integer> result = index.searchNearestNeighbor(query, k);
      time = System.currentTimeMillis() - time;
      System.out.println("\tSFATree:" + (time / 1000.0) + "s");

      List<Double> distances = result.keys();
      //System.out.println("\tTS seen: " + index.getTimeSeriesRead());
      //System.out.println("\tLeaves seen " + index.getIoTimeSeriesRead());
      //System.out.println("\tNodes seen " + index.getBlockRead());
      index.resetIoCosts();

      // compare with nearest neighbor search!
      time = System.currentTimeMillis();
      double resultDistance = Double.MAX_VALUE;
      for (int ww = 0; ww < size; ww++) { // faster than reevaluation in for
        // loop
        double distance = SFATrieTest.getEuclideanDistance(timeSeries, query, means[ww], stds[ww], resultDistance, ww);
        resultDistance = Math.min(distance, resultDistance);
      }
      time = System.currentTimeMillis() - time;
      System.out.println("\tEuclidean:" + (time / 1000.0) + "s");

      Assert.assertEquals("Distances do not match: " + resultDistance + "\t" + distances.get(0),
          distances.get(0), resultDistance, 0.003);
    }

    System.out.println("All ok...");
  }

  /**
   * Builds one SFA trie for each bucket and merges these to one SFA trie.
   *
   * @param l
   * @param leafThreshold
   * @param windowLength
   * @param trieDepth
   * @param sfa
   * @return
   */
  protected SFATrie buildSFATrie(int l, int leafThreshold, int windowLength, int trieDepth, SFA sfa) {
    long time;// now, process each bucket on disk
    SFATrie index = null;

    System.out.println("Building and merging Trees:");
    File directory = tempDir;

    // create an index for each bucket and merge the indices
    for (File bucket : directory.listFiles()) {
      if (bucket.isFile() && bucket.getName().contains("bucket")) {
        time = System.currentTimeMillis();
        List<SFATrie.Approximation[]> windows = readFromFile(bucket);
        if (!windows.isEmpty()) {
          SFATrie trie = new SFATrie(l, leafThreshold, sfa);
          trie.buildIndex(windows, trieDepth);

          if (index == null) {
            index = trie;
          } else {
            index.mergeTrees(trie);
          }

          System.out.println("Merging done in " + (System.currentTimeMillis() - time) + " ms. " + "\t Elements: "
              + index.getSize() + "\t Height: " + index.getHeight());
        }
      }
    }

    // path compression
    if (index != null) {
      index.compress(true);
    }

    // write index to disk?
    // System.out.println("Writing index to disk...");
    // File location = new File("./tmp/sfatrie.idx");
    // location.deleteOnExit();
    // index.writeToDisk(location);

    return index;
  }

  public void performGC() {
    try {
      System.gc();
      Thread.sleep(10);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
  }

  /**
   * Reads a bucket with Fourier-approximations from a file
   *
   * @param name
   * @return
   */
  protected List<SFATrie.Approximation[]> readFromFile(File name) {
    System.out.println("Reading from : " + name.toString());
    long count = 0;
    List<SFATrie.Approximation[]> data = new ArrayList<>();
    try (ObjectInputStream in = new ObjectInputStream(new BufferedInputStream(new FileInputStream(name)))) {
      SFATrie.Approximation[] d = null;
      while ((d = (SFATrie.Approximation[]) in.readObject()) != null) {
        data.add(d);
        count += d.length;
      }
    } catch (EOFException e) {
      // ignore EOFException
    } catch (Exception e) {
      Assert.fail(e.getMessage());
    }
    System.out.println("\t" + count + " time series read.");

    return data;
  }

  /**
   * Opens multiple streams to disk but writes them sequentially to disk
   *
   */
  static class SerializedStreams {
    LinkedBlockingQueue<SFATrie.Approximation>[] wordPartitions;
    ObjectOutputStream[] partitionsStream;

    // the number of TS until the array is written to disk
    static final int minWriteToDiskLimit = 100000;

    long[] writtenSamples;
    long totalBytes = 0;
    double time = 0;

    @SuppressWarnings("unchecked")
    public SerializedStreams(final int useLetters) {
      // the number of partitions to process
      int count = (int) Math.pow(SFATrie.symbols, useLetters);
      this.wordPartitions = new LinkedBlockingQueue[count];
      this.partitionsStream = new ObjectOutputStream[count];

      this.writtenSamples = new long[count];
      this.time = System.currentTimeMillis();

      for (int i = 0; i < this.wordPartitions.length; i++) {
        this.wordPartitions[i] = new LinkedBlockingQueue<>(minWriteToDiskLimit * 2);
        this.writtenSamples[i] = 0;
      }
    }

    /**
     * Close all file streams
     */
    public void setFinished() {
      // finish all data
      for (int i = 0; i < SerializedStreams.this.wordPartitions.length; i++) {
        try {
          // copy contents to current and write these to disk
          List<SFATrie.Approximation> current = new ArrayList<>(this.wordPartitions[i].size());
          this.wordPartitions[i].drainTo(current);
          writeToDisk(current, i);
        } catch (Exception e) {
          Assert.fail(e.getMessage());
        }
      }
      // wait for all futures/threads to finish
      while (!futures.isEmpty()) {
        try {
          futures.remove().get();
        } catch (Exception e) {
          Assert.fail(e.toString());
        }
      }
      // close all streams
      long totalTSwritten = 0;
      for (int i = 0; i < SerializedStreams.this.wordPartitions.length; i++) {
        try {
          if (partitionsStream[i] != null) {
            partitionsStream[i].close();
            totalTSwritten += writtenSamples[i];
          }
        } catch (Exception e) {
          Assert.fail(e.getMessage());
        }
      }
      System.out.println("Time series written:" + totalTSwritten);
    }

    /**
     * Adds a time series to the corresponding queue and writes the bucket to
     * disk if the bucket exceeds minWriteToDiskLimit elements
     */
    public void addToPartition(byte[] words, double[] data, int offset, int prefixLength) {
      try {
        // the bucket
        final int l = getPrefix(words, prefixLength);
        this.wordPartitions[l].put(new SFATrie.Approximation(data, words, offset));

        // write to disk
        synchronized (this.wordPartitions[l]) {
          if (this.wordPartitions[l].size() >= minWriteToDiskLimit) {
            // copy the elements to current and write this to disk
            final List<SFATrie.Approximation> current = new ArrayList<>(this.wordPartitions[l].size());
            this.wordPartitions[l].drainTo(current);
            futures.add(serializerExec.submit(new Callable<Long>() {
              @Override
              public Long call() throws Exception {
                writeToDisk(current, l);
                totalBytes += current.size() * 20 * 8;
                return totalBytes;
              }
            }));
          }

        }
      } catch (Exception e) {
        Assert.fail(e.getMessage());
      }
    }

    /**
     * Gets a prefix of queryLength useLetters from the word, encoded as int.
     *
     * @param word
     * @param useLetters
     * @return
     */
    protected int getPrefix(byte[] word, int useLetters) {
      int id = word[0];
      if (useLetters > 1) {
        id = id * SFATrie.symbols + word[1];
      }
      if (useLetters > 2) {
        id = id * SFATrie.symbols + word[2];
      }
      return id;
    }

    /**
     * Serializes the bucket to disk
     *
     * @param current
     * @param letter
     * @throws FileNotFoundException
     * @throws IOException
     */
    protected void writeToDisk(List<SFATrie.Approximation> current, int letter) throws IOException {
      if (!current.isEmpty()) {
        if (partitionsStream[letter] == null) {
          String fileName = tempDir.getAbsolutePath() + File.separator + letter + ".bucket";
          File file = new File(fileName);
          file.deleteOnExit();
          partitionsStream[letter] = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(file, false),
              1048576 * 8 /* 8mb */));
        }

        partitionsStream[letter].writeUnshared(current.toArray(new SFATrie.Approximation[]{}));

        // reset the references to the objects to allow
        // the garbage collector to write the objects
        partitionsStream[letter].reset();

        try {
          Thread.sleep(100);
        } catch (InterruptedException e) {
          Assert.fail(e.getMessage());
        }

        this.writtenSamples[letter] += current.size();
      }
    }
  }

  @After
  public void tearDown() throws Exception {
    serializerExec.shutdown();
    transformExec.shutdown();
  }
}