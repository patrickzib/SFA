// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.index;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Ordered List of Tuples with a maximum of <code>maxSize</code> Elements
 */
public class SortedListMap<K extends Comparable<K>, V> {

  private List<K> keys = new ArrayList<>();
  private List<V> values = new ArrayList<>();
  private int maxSize = 0;
  private final static int UNDEFINED = -1;

  public SortedListMap(final int maxSize) {
    this.maxSize = maxSize;
  }

  public SortedListMap() {
    this.maxSize = UNDEFINED;
  }

  public void clear() {
    this.keys = new ArrayList<>();
    this.values = new ArrayList<>();
  }

  public boolean containsKey(K key) {
    return Collections.binarySearch(this.keys, key) >= 0;
  }

  public void replaceValue(K key, V value) {
    int pos = findFirstOccurrence(key);
    if (pos >= 0) {
      this.values.set(pos, value);
    }
  }

  private int findFirstOccurrence(K key) {
    int pos = Collections.binarySearch(this.keys, key);
    if (pos > 0) {
      while (pos > 0 && this.keys.get(pos - 1).equals(key)) {
        pos--;
      }
    }
    return pos;
  }

  private int findLastOccurrence(K key) {
    int pos = Collections.binarySearch(this.keys, key);
    if (pos > 0) {
      while (pos < this.keys.size() - 1 && this.keys.get(pos + 1).equals(key)) {
        pos++;
      }
    }
    return pos;
  }

  public V getFirstOccurrence(K key) {
    int pos = findFirstOccurrence(key);
    if (pos >= 0) {
      return this.values.get(pos);
    }
    throw new NoSuchElementException("Element with " + key + " not present.");
  }

  public V getLastOccurrence(K key) {
    int pos = findLastOccurrence(key);
    if (pos >= 0) {
      return this.values.get(pos);
    }
    throw new NoSuchElementException("Element with " + key + " not present.");
  }

  public void put(K key, V value) {
    int pos = findFirstOccurrence(key);

    if (pos >= 0 && this.values.get(pos) != value) {
      this.keys.add(pos, key);
      this.values.add(pos, value);
    }

    if (pos < 0) {
      this.keys.add(-pos - 1, key);
      this.values.add(-pos - 1, value);
    }

    if (size() > this.maxSize && this.maxSize != UNDEFINED) {
      this.keys.remove(this.keys.size() - 1);
      this.values.remove(this.values.size() - 1);
    }
  }

  /**
   * Removes the first occurrence of 'key'
   *
   * @param key
   * @return
   */
  public V removeFirst(K key) {
    int pos = findFirstOccurrence(key);
    if (pos >= 0) {
      this.keys.remove(pos);
      return this.values.remove(pos);
    }
    throw new NoSuchElementException("Element with " + key + " not present.");
  }

  public int size() {
    return this.keys.size();
  }

  public List<K> keys() {
    return this.keys;
  }

  public List<V> values() {
    return this.values;
  }

  public K firstKey() {
    return this.keys.get(0);
  }

  public K lastKey() {
    return this.keys.get(this.keys.size() - 1);
  }

  public boolean isEmpty() {
    return this.keys.isEmpty();
  }
}
