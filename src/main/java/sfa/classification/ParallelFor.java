// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.classification;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A class for parallel execution of tasks
 */
public class ParallelFor {

  private static int CPUs = Math.min(Runtime.getRuntime().availableProcessors(), 40);
  private static ExecutorService executor = Executors.newFixedThreadPool(CPUs);

  public interface Each {
    void run(int i, AtomicInteger processed);
  }

  public static int withIndex(ExecutorService executor, final int chunksize, final Each body) {
    final CountDownLatch latch = new CountDownLatch(chunksize);
    final AtomicInteger processed = new AtomicInteger(0);
    LinkedList<Future<?>> futures = new LinkedList<>();

    for (int i = 0; i < chunksize; i++) {
      final int ii = i;
      futures.add(executor.submit(new Runnable() {
        public void run() {
          try {
            body.run(ii, processed);
          } catch (Exception e) {
            e.printStackTrace();
          } finally {
            latch.countDown();
          }
        }
      }));
    }
    try {
      latch.await();

      while (!futures.isEmpty()) {
        futures.removeFirst().get();
      }

    } catch (InterruptedException e) {
      e.printStackTrace();
      //executor.shutdownNow();
    } catch (Exception e) {
      e.printStackTrace();
    }
//    finally {
//      executor.shutdown();
//    }
    return processed.get();
  }

  public static int withIndex(int stop, final Each body) {
    return withIndex(executor, stop, body);
  }

  public static void shutdown() {
    executor.shutdown();
  }
}
