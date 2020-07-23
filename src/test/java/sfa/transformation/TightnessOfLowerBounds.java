package sfa.transformation;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import sfa.SFAWordsTest;
import sfa.classification.ParallelFor;
import sfa.index.SFATrie;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

public class TightnessOfLowerBounds {

  // number of coefficients for each representation
  public static int startCoeff = 8;
  public static int endCoeff = 256;
  public static int tslength = 256;
  public final static int count = 1000;

  public static int finalCoeff = (int)((Math.log(endCoeff)-Math.log(startCoeff))/Math.log(2.0))+1;

  static Representation[] representations = new Representation[] {
    new APCA(),
    new DFT(),
    new DWT(),
    new PAA(),
    new PLA()
  };

  static String[] datasets = new String[]{
      "steamgen.dat",
      "PostureCentroidB",
      "winding.dat",
      "ann_gun_CentroidA",
      "power_data.dat",
//      "muscle_activation.dat",  // not uploaded
//      "Fluid_dynamics.dat",
//      "buoy_sensor.dat",
//      "burst.dat",
//      "cstr.dat",
//      "foetal_ecg.dat",
//      "tide.dat",
//      "3_7_2006_trajectory.txt",
//      "mitdbx_mitdbx_108.txt",
//      "chfdbchf15.txt",
//      "nprs43.txt",
//      "TOR97.DAT"
  };


  public static void main (String[] argv) throws IOException {
    try {
      System.out.println("Start coefficients: " + startCoeff);
      System.out.println("End coefficients: " + endCoeff);
      System.out.println("tslength: " + tslength);

      ClassLoader classLoader = SFAWordsTest.class.getClassLoader();
      File dir = new File(classLoader.getResource("datasets/indexing/").getFile());

      // read time series datasets
      final double[][][] tightnessOfLowerBounds
          = new double[datasets.length]
                      [representations.length]
                      [finalCoeff];

      for (int d = 0; d < datasets.length; d++) {
        final int dataD = d;
        System.out.println("\n"+datasets[d]);
        List<TimeSeries> samples = Arrays.asList(
            TimeSeriesLoader.readSampleSubsequence(
                new File(dir.getAbsolutePath()+ "/" +datasets[d])).getSubsequences(tslength, true));
        System.out.println("Samples " + samples.size());

        if (samples.size() > 2*count) {
          final TimeSeries[] samples1 = samples.subList(0, count).toArray(new TimeSeries[]{});
          final TimeSeries[] samples2 = samples.subList(count,2*count).toArray(new TimeSeries[]{});
        	final double[] trueDist = new double[samples1.length];

          // True Euclidean distance
          for (int q = 0; q < samples1.length; q++) {
            trueDist[q] = getEuclideanDistance(samples1[q], samples2[q]);
          }

          // Iterate all representations and compute lower bounding distance
          ParallelFor.withIndex(representations.length, new ParallelFor.Each() {
            @Override
            public void run(int id, AtomicInteger processed) {
              for (int i = 0; i < representations.length; i++) {
                if (id==i) {
                  Representation representation = representations[i];

            	    for (int c = startCoeff, aa=0; c <= endCoeff; c*=2, aa++) {
              	    // compute representations
              	    TimeSeries[][] transformedSamples = new TimeSeries[2][samples1.length];
              	    for (int a = 0; a < samples1.length; a++) {
                      transformedSamples[0][a] = representation.transform(samples1[a], c);
                      transformedSamples[1][a] = representation.transform(samples2[a], c);
                      //transformedSamples[1][a] = samples2[a];
                    }

                		for (int q = 0; q < samples1.length; q++) {
                			double dist = representation.getDistance(
                			    transformedSamples[0][q],
                          transformedSamples[1][q],
                          samples2[q],
                          tslength,
                          Double.MAX_VALUE);

                			// lower bounding distance / euclidean distance
                			if (trueDist[q] > 0.0001) {
                			  double value = dist / trueDist[q];
                			  if (!Double.isNaN(value)) {
                			    tightnessOfLowerBounds[dataD][i][aa] += value;
                			  }
                			}
                			else {
                				tightnessOfLowerBounds[dataD][i][aa] += 1;
                			}

                			if (dist-trueDist[q] > 0.0001) {
                			  System.out.println(
                			      "Error: "+representation.getClass().getSimpleName()+" distance "
                                + dist + " is larger than true dist:" + trueDist[q]);
                			}
                    }
            	    }
                }
              }
            }
          });

          System.out.println("");
          System.out.println("\tTightness of Lower Bounds: ");

          System.out.print("l");
          for (int i=0; i < representations.length; i++) {
            Representation signature = representations[i];
            System.out.print("\t"+signature.getClass().getSimpleName());
          }
          System.out.println();

          for (int c = startCoeff, a = 0; c <= endCoeff; c*=2, a++) {
            System.out.print(c);
            for (int i=0; i < representations.length; i++) {
              System.out.print("\t"+(Math.round(1000.0*tightnessOfLowerBounds[d][i][a] / (double)(count))/1000.0));
            }
            System.out.println("");
          }
        }
      }

      System.out.println("");
      System.out.println("Total Averages");
      System.out.println("");

      getTightness(tightnessOfLowerBounds);
    } finally {
      ParallelFor.shutdown();
    }
  }

  /**
   * Basic implementation of the Euclidean Distance
   */
  public static double getEuclideanDistance (TimeSeries t1, TimeSeries t2) {
    final int sizeT1 = t1.getLength();

    double distance = 0;
    double[] t1Values = t1.getData();
    double[] t2Values = t2.getData();

    for (int i = 0; i < Math.min(sizeT1, t2.getLength()); i++) {
      double value = t1Values[i] - t2Values[i];
      distance += value * value;
    }

    return distance;
  }

  private static void getTightness(double[][][] tightnessOfLowerBounds) {

    double[][] avgTightnessOfLowerBounds = new double[tightnessOfLowerBounds[0].length][tightnessOfLowerBounds[0][0].length];

    for (double[][] tightnessOfLowerBound : tightnessOfLowerBounds) {
      for (int i=0; i < tightnessOfLowerBounds[0].length; i++) {
        for (int c = 0; c < tightnessOfLowerBounds[0][0].length; c++) {
          avgTightnessOfLowerBounds[i][c] += tightnessOfLowerBound[i][c] / (double)(count);
        }
      }
    }

    System.out.print("l");
    for (int i=0; i < representations.length; i++) {
      System.out.print("\t"+representations[i].getClass().getSimpleName());
    }
    System.out.println();

    for (int c = 0; c < tightnessOfLowerBounds[0][0].length; c++) {
      System.out.print(c);
      for (int i=0; i < representations.length; i++) {
        System.out.print("\t"+(Math.round(1000.0*avgTightnessOfLowerBounds[i][c] / tightnessOfLowerBounds.length)/1000.0));
      }
      System.out.println("");
    }


    System.out.println("");
    System.out.println("Highest TLB per dataset");
    System.out.println("");
    System.out.print("ds");
    for (int i=0; i < representations.length; i++) {
      System.out.print("\t"+representations[i].getClass().getSimpleName());
    }
    System.out.println();

    for (int c = 0; c < datasets.length; c++) {
      System.out.print(datasets[c]);
      for (int i=0; i < representations.length; i++) {
        System.out.print("\t"+(Math.round(1000.0*tightnessOfLowerBounds[c][i][finalCoeff-1] / (double)(count))/1000.0));
      }
      System.out.println("");
    }


  }

}
