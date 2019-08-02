// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.timeseries;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import com.carrotsearch.hppc.DoubleArrayList;

public class TimeSeriesLoader {

  /**
   * Loads the time series from a csv-file of the UCR time series archive.
   *
   * @param dataset
   * @return
   */
  public static TimeSeries[] loadDataset(String dataset) {
    return loadDataset(new File(dataset));
  }

  /**
   * Loads the time series from a csv-file of the UCR time series archive.
   *
   * @param dataset
   * @return
   */
  public static TimeSeries[] loadDataset(File dataset) {
    ArrayList<TimeSeries> samples = new ArrayList<>();

    try (BufferedReader br = new BufferedReader(new FileReader(dataset))) {
      String line = null;
      while ((line = br.readLine()) != null) {
        if (line.startsWith("@")) {
          continue;
        }

        // switch between old " " and new separator "," in the UCR archive
        String separator = (line.contains(",") ? "," : " ");
        String[] columns = line.split(separator);

        double[] data = new double[columns.length];
        int j = 0;
        Double label = null;

        // first is the label
        int i = 0;
        for (; i < columns.length; i++) {
          String column = columns[i].trim();
          if (isNonEmptyColumn(column)) {
            label = Double.valueOf(column);
            break;
          }
        }

        // next the data
        for (i = i + 1; i < columns.length; i++) {
          String column = columns[i].trim();
          try {
            if (isNonEmptyColumn(column)) {
              data[j++] = Double.parseDouble(column);
            }
          } catch (NumberFormatException nfe) {
            nfe.printStackTrace();
          }
        }
        if (j > 0) {
          TimeSeries ts = new TimeSeries(Arrays.copyOfRange(data, 0, j), label);
          ts.norm();
          samples.add(ts);
        }
      }

    } catch (IOException e) {
      e.printStackTrace();
    }

    System.out.println("Done reading from " + dataset.getName() + " samples " + samples.size() + " queryLength " + samples.get(0).getLength());
    return samples.toArray(new TimeSeries[]{});
  }

  public static MultiVariateTimeSeries[] loadMultivariateDatset(
      File dataset, boolean derivatives) throws IOException {

    List<MultiVariateTimeSeries> samples = new ArrayList<>();
    List<Double>[] mts = null;
    int lastId = -1;

    try (BufferedReader br = new BufferedReader(new FileReader(dataset))) {
      String line = null;
      double label = -1;
      while ((line = br.readLine()) != null) {
        String[] columns = line.split(" ");

        // id
        int id = Integer.valueOf(columns[0].trim());
        if (id != lastId) {
          addMTS(samples, mts, label);

          // initialize
          lastId = id;
          mts = new ArrayList[columns.length - 3];
          for (int i = 0; i < mts.length; i++) {
            mts[i] = new ArrayList<>();
          }
          // label
          label = Double.valueOf(columns[2].trim());
        }

        // timeStamp
        /* int timeStamp = Integer.valueOf(columns[1].trim()); */

        // the data
        for (int dim = 0; dim < columns.length - 3; dim++) { // all dimensions
          String column = columns[dim + 3].trim();
          try {
            double d = Double.parseDouble(column);
            mts[dim].add(d);
          } catch (NumberFormatException nfe) {
            nfe.printStackTrace();
          }
        }
      }

      addMTS(samples, mts, label);
    } catch (IOException e) {
      e.printStackTrace();
    }

    System.out.println("Done reading from " + dataset
        + " samples " + samples.size()
        + " dimensions " + samples.get(0).getDimensions());

    MultiVariateTimeSeries[] m = samples.toArray(new MultiVariateTimeSeries[]{});
    return (derivatives)? getDerivatives(m) : m;
  }

  protected static MultiVariateTimeSeries[] getDerivatives(MultiVariateTimeSeries[] mtsSamples) {
    for (MultiVariateTimeSeries mts : mtsSamples) {
      TimeSeries[] deltas = new TimeSeries[2 * mts.timeSeries.length];
      TimeSeries[] samples = mts.timeSeries;
      for (int a = 0; a < samples.length; a++) {
        TimeSeries s = samples[a];
        double[] d = new double[s.getLength()];
        for (int i = 1; i < s.getLength(); i++) {
          d[i - 1] = Math.abs(s.getData()[i] - s.getData()[i - 1]);
        }
        deltas[a] = samples[a];
        deltas[mts.timeSeries.length + a] = new TimeSeries(d, mts.getLabel());
      }
      mts.timeSeries = deltas;
    }
    return mtsSamples;
  }

  protected static void addMTS(List<MultiVariateTimeSeries> samples, List<Double>[] mts, double label) {
    if (mts != null && mts[0].size() > 0) {
      TimeSeries[] dimensions = new TimeSeries[mts.length];
      for (int i = 0; i < dimensions.length; i++) {
        double[] rawdata = new double[mts[i].size()];
        int j = 0;
        for (double d : mts[i]) {
          rawdata[j++] = d;
        }
        dimensions[i] = new TimeSeries(rawdata, label);
      }
      samples.add(new MultiVariateTimeSeries(dimensions, label));
    }
  }

  public static TimeSeries readSampleSubsequence(File dataset) throws IOException {
    try (BufferedReader br = new BufferedReader(new FileReader(dataset))) {
      DoubleArrayList data = new DoubleArrayList();
      String line = null;
      while ((line = br.readLine()) != null) {
        line = line.trim();
        String[] values = line.split("[ \\t]");
        if (values.length > 0) {
          for (String value : values) {
            try {
              value = value.trim();
              if (isNonEmptyColumn(value)) {
                data.add(Double.parseDouble(value));
              }
            } catch (NumberFormatException nfe) {
              // Parse-Exception ignored
            }
          }
        }
      }
      return new TimeSeries(data.toArray());
    }
  }

  public static TimeSeries[] readSamplesQuerySeries(String dataset) throws IOException {
    return readSamplesQuerySeries(new File(dataset));
  }

  public static TimeSeries[] readSamplesQuerySeries(File dataset) throws IOException {
    List<TimeSeries> samples = new ArrayList<>();
    try (BufferedReader br = new BufferedReader(new FileReader(dataset))) {
      String line = null;
      while ((line = br.readLine()) != null) {
        DoubleArrayList data = new DoubleArrayList();
        line = line.trim();
        String[] values = line.split("[ \\t]");
        if (values.length > 0) {
          for (String value : values) {
            try {
              value = value.trim();
              if (isNonEmptyColumn(value)) {
                data.add(Double.parseDouble(value));
              }
            } catch (NumberFormatException nfe) {
              // Parse-Exception ignored
            }
          }
          samples.add(new TimeSeries(data.toArray()));
        }
      }
    }
    return samples.toArray(new TimeSeries[]{});
  }

  public static boolean isNonEmptyColumn(String column) {
    return column != null && !"".equals(column) && !"NaN".equals(column) && !"\t".equals(column);
  }

  public static TimeSeries generateRandomWalkData(int maxDimension, Random generator) {
    double[] data = new double[maxDimension];

    // Gaussian Distribution
    data[0] = generator.nextGaussian();

    for (int d = 1; d < maxDimension; d++) {
      data[d] = data[d - 1] + generator.nextGaussian();
    }

    return new TimeSeries(data);
  }
}
