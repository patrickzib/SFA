// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public class ParallelFor {

  public static int CPUs = Runtime.getRuntime().availableProcessors();
  static ExecutorService executor = Executors.newFixedThreadPool(CPUs);

  public interface Each {
    void run(int i, AtomicInteger processed);
  }

  public static int withIndex(ExecutorService executor, int stop, final Each body) {
    int chunksize = stop;
    final CountDownLatch latch = new CountDownLatch(chunksize);
    final AtomicInteger processed = new AtomicInteger(0);

    for (int i=0; i<stop; i++) {
      final int ii = i;
      executor.submit(new Runnable() {
        public void run() {
          try {
            body.run(ii, processed);
          } catch (Exception e) {
            e.printStackTrace();
          } finally {
            latch.countDown();
          }
        }
      });
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      e.printStackTrace();
      executor.shutdownNow();
    }
    finally {
//      executor.shutdown();
    }
    return processed.get();
  }

  public static int withIndex(int stop, final Each body) {
    return withIndex(executor, stop, body);
  }

  public static void shutdown()  {
    executor.shutdown();
  }

  public static void main(String [] argv) {
    ParallelFor.withIndex(4, new ParallelFor.Each() {
      public void run(int i, AtomicInteger processed) {
        System.out.println(i*10);
      }
    });
  }


}
