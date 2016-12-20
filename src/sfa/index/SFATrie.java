// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.index;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.TreeMap;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import sfa.timeseries.TimeSeries;
import sfa.transformation.SFA;
import sfa.transformation.SFA.HistogramType;

/**
 * Implementation of the SFATrie for time series subsequence-search
 * using SFA.
 *
 * See publication:
 *    Schäfer, P., Högqvist, M.: SFA: a symbolic fourier approximation and 
 *    index for similarity search in high dimensional datasets. 
 *    In: EDBT, ACM (2012)
 *
 */
public class SFATrie implements Serializable {
  private static final long serialVersionUID = 8983404060948074333L;

  protected SFANode root;

  protected int wordLength;
  protected int leafThreshold;

  protected int minimalHeight = -1;

  public SFA quantization = null;
  public final static int symbols  = 8;

  protected boolean compressed = false;

  protected transient long ioBlockRead = 0;
  protected transient long ioTimeSeriesRead = 0;
  protected transient long timeSeriesRead = 0;

  public static enum NodeType { Leaf, Internal };

  TimeSeries timeSeries;
  double[] means;
  double[] stddev;
  
  /**
   * Create a new SFATrie with dimenionality l and threshold 'leafThreshold'.
   * A leaf will be split, once leafThreshold is exceeded.
   * @param l word length of the SFA transformation
   * @param leafThreshold number of ts in a leaf
   * @param quantization
   */
  public SFATrie(int l, int leafThreshold) {
    this(l, leafThreshold, new SFA(HistogramType.EQUI_FREQUENCY));
  }
  
  public SFATrie(int l, int leafThreshold, SFA quantization) {
    this.quantization = quantization;

    this.wordLength = l;

    this.root = new SFANode(new byte[0], symbols, this.wordLength);
    this.leafThreshold = leafThreshold;

    resetIoCosts();
  }
  
  /**
   * Calculates the mean and Stddev for the particular time series
   * @param samples
   * @param windowLength
   * @param idx
   */
  public void caluclateMeanStddev(int windowLength) {
    int size = (this.timeSeries.getData().length-windowLength)+1;
    this.means = new double[size];
    this.stddev = new double[size];
    TimeSeries.calcIncreamentalMeanStddev(windowLength, this.timeSeries, this.means, this.stddev);
  }
  
//  /**
//   * Inserts multiple given time series
//   * TODO dont mix whole matching and subsequence matching!
//   */
//  public void buildIndex(TimeSeries[] samples, int windowLength) {
//    // Train the SFA quantization histogram
//    this.quantization.fitWindowing(samples, windowLength, this.wordLength, symbols, true, true);
//    
//    // calculates means and stddev
//    setTimeSeries(samples, windowLength);
//    
//    // Transform the time series to SFA words
//    for (int i = 0; i < samples.length; i++) {
//      TimeSeries ts = samples[i];
//      double[][] words = this.quantization.transformWindowingDouble(ts, this.wordLength);
//    
//      // insert each timeseries window
//      for (int offset = 0; offset < words.length; offset++) {
//        TimeSeriesWindow window = new TimeSeriesWindow(
//            words[offset], 
//            this.quantization.quantizationByte(words[offset]),
//            i,
//            offset);
//        insert(window, 0, this.root);
//      }
//    }
//
//    printStats(true);
//  }
  
  /**
   * Inserts a given time series
   *
   */
  public void buildIndex(TimeSeries ts, int windowLength) {
    // Train the SFA quantization histogram
    this.quantization.fitWindowing(new TimeSeries[]{ts}, windowLength, this.wordLength, symbols, true, true);

    // calculates means and stddev
    setTimeSeries(ts, windowLength);
    
    // Transform the time series to SFA words
    double[][] words = this.quantization.transformWindowingDouble(ts, this.wordLength);

    // insert each timeseries window
    for (int offset = 0; offset < words.length; offset++) {
      TimeSeriesWindow window = new TimeSeriesWindow(
          words[offset], 
          this.quantization.quantizationByte(words[offset]),
          offset);
      insert(window, 0, this.root);
    }

    
    printStats(true);
  }

  /**
   * Bulk insertion into the SFA trie
   */
  public void buildIndex(List<SFATrie.TimeSeriesWindow[]> windows, int minHeight, int windowLength) {
    this.minimalHeight = minHeight; // TODO move to constructor??
    
    // insert each timeseries window
    for (SFATrie.TimeSeriesWindow[] w : windows) {
      for (SFATrie.TimeSeriesWindow window : w) {
        insert(window, 0, this.root);
      }
    }
   
    printStats(true);
  }

  public void printStats(boolean compress) {
    if (!compress) {
      System.out.println("\tLeaves (not path-compressed): " + getLeafCount());
      System.out.println("\tHeight " + getHeight());
      System.out.println("\tNodes " + getNodeCount());
      System.out.println("\tElements " + getSize());
    }    
    else if (compress) {
      compress();
  
      System.out.println("\tLeaves (path-compressed): " + getLeafCount());
      System.out.println("\tHeight " + getHeight());
      System.out.println("\tNodes " + getNodeCount());
      System.out.println("\tElements " + getSize());
    }
  }
  
  private void insert(
      SFANode nodeToInsert,
      byte[] path,
      int index,
      SFANode node,
      SFANode parentNode) {

    // adapt min-max-bounds
    node.adaptMinMaxValues(nodeToInsert);

    if (node.type == NodeType.Internal) {
      byte key = path[index];
      SFANode childNode = node.getChild(key);

      if (childNode != null) {
        // iterate further
        if (index + 1 < path.length) {
          insert(nodeToInsert, path, index + 1, childNode, node);
        } else {
          // add all child nodes
          if (nodeToInsert.type == NodeType.Internal) {
            for (SFANode child : nodeToInsert.children) {
              if (child != null) {
                insert(child, child.word, index + 1, childNode, node);
              }
            }
            nodeToInsert.children = null;
          }
          // insert all time series
          else if (nodeToInsert.type == NodeType.Leaf) {
            for (TimeSeriesWindow ts : nodeToInsert.getElements()) {
              insert(ts, index, node);
            }
            nodeToInsert.removeElements();
          }
        }
      } else {
        // append the new internal node
        node.addChild(key, nodeToInsert);
      }
    } else if (node.type == NodeType.Leaf) {
      if (nodeToInsert.type == NodeType.Internal) {
        // extend node by length
        node.type = NodeType.Internal;
        for (TimeSeriesWindow ts : node.getElements()) {
          insert(ts, index, node);
        }
        node.removeElements();

        // add new nodeToInsert again
        insert(nodeToInsert, path, index, node, parentNode);
      } else if (nodeToInsert.type == NodeType.Leaf) {
        // insert all time series
        // append all elements from node into nodeToInsert
        for (TimeSeriesWindow ts : nodeToInsert.getElements()) {
          insert(ts, index - 1, parentNode);
        }
        nodeToInsert.removeElements();
      }
    }

  }

  /**
   * Inserts the given element at the given path.
   *
   * @param path
   *          the path where to insert the element.
   * @param element
   *          the element to be inserted.
   */
  protected void insert(
      TimeSeriesWindow element,
      int index,
      SFANode node) {

    // adapt min-max-bounds
    node.adaptMinMaxValues(element);

    byte key = element.word[index];
    SFANode childNode = node.getChild(key);

    if (childNode == null) {
      // add a new child node
      childNode = node.addChild(key, symbols, this.wordLength);

      // if needed, guarantee a minimal height (used for bulk loading)
      if (this.minimalHeight - 1 > index) {
        childNode.type = NodeType.Internal;
        insert(element, index + 1, childNode);
      }
      // else, just insert the time series
      else {
        childNode.addElement(element);
      }
      return;
    }

    // internal node
    if (childNode.type == NodeType.Internal) {
      insert(element, index + 1, childNode);
    }
    // insert into the leaf if it still fits
    // else split the node and reinsert all timeseries
    else if (childNode.type == NodeType.Leaf) {
      if (childNode.getSize() < this.leafThreshold) {
        childNode.addElement(element);
      }
      // last element was inserted
      else if (index >= element.word.length - 1) {
        childNode.addElement(element);
      }
      // split leaf
      else {
        childNode.type = NodeType.Internal;

        // reinsert all elements into the trie
        for (TimeSeriesWindow ts : childNode.getElements()) {
          insert(ts, index + 1, childNode);
        }

        // add the new elements
        insert(element, index + 1, childNode);

        // remove elements from old child node
        childNode.removeElements();
      }
    }
  }

  /**
   * Merge two trees
   *
   * @param t
   */
  public void mergeTrees(SFATrie tree) {
    // test if children are not contained
    // insert children of root node
    for (SFANode node : tree.root.children) {
      if (node != null) {
        insert(node, node.getWord(), 0, this.root, this.root);
      }
    }
    if (tree.root.getElements() != null && !tree.root.getElements().isEmpty()) {
      throw new RuntimeException("error");
    }
    tree.root = null;
    
    this.compressed = false;
  }

  public TimeSeries getTimeSeries() {
    return timeSeries;
  }

  public void setTimeSeries(TimeSeries ts, int windowLength) {
    this.timeSeries = ts;    
    caluclateMeanStddev(windowLength);
  }
  
  /**
   * Applies path compression of on the SFA try
   * @param m
   */
  public void compress() {
    if (!this.compressed) {
      compress(this.root);
      this.compressed = true;
    }
  }

  protected void compress(SFANode m) {
    SFANode node = (SFANode) m;
    // join adjacent nodes
    if (node.type == NodeType.Internal) {
      SFANode previousNode = null;
//      int count = 0;
      for (int i = 0; i < node.children.length; i++) {
        SFANode currentNode = node.children[i];
        if (currentNode != null) {
          // leaf node
          if (currentNode.type == NodeType.Leaf) {
            // clean up currentNode's ts
            for (TimeSeriesWindow ts : currentNode.getElements()) {
              ts.fourierValues = null;
              ts.word = null;                
            }
            if (previousNode != null
                && previousNode != currentNode
                && (previousNode.getSize()
                    + currentNode.getSize() < this.leafThreshold)) {

              // merge nodes
              previousNode.addAll(currentNode.getElements());
              previousNode.adaptMinMaxValues(currentNode);
              currentNode.removeElements();
              node.children[i] = previousNode;
            } else {
              previousNode = currentNode;
            }
          }
          // Internal node
          else {
            previousNode = null;
            compress(currentNode);
          }
        }
//        // count non-empty positions
//        count++;
      }

//      // remove unnecessary branches
//      SFANode[] children = node.children;
//      node.children = new SFANode[count];
//      for (int i = 0, a = 0; i < children.length; i++) {
//        if (children[i] != null) {
//          node.children[a++] = children[i];
//        }
//      }
    }
  }

  /**
   * Retrieves the leaf node to 'path'. If there is no
   * exact leaf node, it returns any sibling node on the path.
   *
   * @param path
   * @return
   */
  public SFANode getLeafNode(byte[] path) {
    SFANode currentNode = this.root;

    for (byte element : path) {
      if (currentNode.type == NodeType.Internal) {
        addToBlockRead(1);
        SFANode newCurrentNode = currentNode.getChild(element);
        if (newCurrentNode == null) {
          // choose abitrary node
          newCurrentNode = currentNode.getChildren().iterator().next();
        }
        currentNode = newCurrentNode;
      } else {
        return currentNode;
      }
    }
    return currentNode;
  }

  /**
   * Approximate search for the query.
   */
  public SortedListMap<Double, TimeSeriesWindow> search(
      double[] dftQuery, byte[] wordQuery, TimeSeries query, int k) {
    SortedListMap<Double, TimeSeriesWindow> result = new SortedListMap<Double, TimeSeriesWindow>(k);

    // search for the exact path
    SFANode node = getLeafNode(wordQuery);

    // leaf node
    if (node != null && node.type == NodeType.Leaf) {
      addToIOTimeSeriesRead(1);
      addToTimeSeriesRead(node.getSize());

      // retrieve all time series
      for (TimeSeriesWindow object : node.getElements()) {
        double originalDistance = getEuclideanDistance(
            timeSeries,
            query,
            means[object.pos],
            stddev[object.pos],
            Double.MAX_VALUE,
            object.pos);
        result.put(originalDistance, object);
      }
      return result;
    } else {
      throw new RuntimeException("Keinen Pfad gefunden!");
    }
  }

  public int getHeight() {
    return this.root.getDepth();
  }

  public int getSize() {
    return this.root.getTotalSize();
  }

  public int getNodeCount() {
    return this.root.getNodeCount();
  }

  public long getLeafCount() {
    return this.root.getLeafCount();
  }

  public ArrayList<SFANode> getLeafNodes() {
    return this.root.getLeafNodes();
  }

  public ArrayList<Integer> getLeafNodeCounts() {
    ArrayList<Integer> leaves = new ArrayList<Integer>();
    return this.root.getLeafNodeCounts(leaves);
  }

  protected SFANode getNode(byte[] path, double epsilonSquare, double error) {
    SFANode currentNode = this.root;

    for (byte element : path) {
      currentNode = currentNode.getChild(element);
      if (currentNode == null) {
        return null;
      }
    }

    return currentNode;

  }

  public SortedListMap<Double, Integer> searchNearestNeighbor(TimeSeries query, int k) {
    // approximation
    double[] dftQuery = quantization.transformation.transform(query, wordLength);
    
    // quantization
    byte[] wordQuery = quantization.quantizationByte(dftQuery);
    
    return searchKNN(dftQuery, wordQuery, query, k);
  }
  
  public SortedListMap<Double, Integer> searchKNN(
      double[] dftQuery, byte[] wordQuery, TimeSeries query, int k) {

    // priority queues ordered by ascending distances
    TreeMap<Double, List<SFANode>> queue = new TreeMap<>();
    SortedListMap<Double, Integer> result = new SortedListMap<>(k); // = search(transformedQuery, query, k);

    // add the root to the branch list
    addToMultiMap(queue, this.root, 0.0);

    while (!queue.isEmpty()) {
      // retrieve first element
      double lbDistance = queue.firstKey();
      SFANode currentNode = removeFromMultiMap(queue, lbDistance);

      double kthBestDistance = (result.size() < k ? Double.MAX_VALUE : result.lastKey());

      if (lbDistance < kthBestDistance) {
        // iterate over all nodes of the trie
        if (currentNode.type == NodeType.Internal) {
          addToBlockRead(1);
          // get distance of the path to the query
          for (SFANode child : currentNode.getChildren()) {
            double distance = getLowerBoundingDistance(
                dftQuery,
                wordQuery,
                child.minValues,
                child.maxValues);
            if (distance < kthBestDistance) {
              addToMultiMap(queue, child, distance);
            }
          }
        }
        // get ED time series in the leaf node
        else {
          addToIOTimeSeriesRead(1);
          addToTimeSeriesRead(currentNode.getSize());

          for (TimeSeriesWindow object : currentNode.getElements()) {
            kthBestDistance = (result.size() < k ? Double.MAX_VALUE : result.lastKey());
            double distance = getEuclideanDistance(
                timeSeries,
                query,
                means[object.pos],
                stddev[object.pos],
                kthBestDistance,
                object.pos);
            if (distance <= kthBestDistance) {
              result.put(distance, object.pos);
            }
          }
        }
      }
      else {
        break;
      }
    }

    return result;
  }

  /**
   * Euclidean distance between a window in ts and the query q
   */
  protected double getEuclideanDistance(
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

  protected double getLowerBoundingDistance(
      double[] dftQuery,
      byte[] wordQuery,
      double[] minValues,
      double[] maxValues) {

    // add distance of the current letter to total distance
    double newDistance = 0.0;
    double[] data = dftQuery;

    for (int i = 0; i < minValues.length; i++) {
      // below
      if (data[i] < minValues[i]) {
        newDistance += getDistance(minValues[i], dftQuery[i]);
      }
      // above
      else if (data[i] > maxValues[i]) {
        newDistance += getDistance(maxValues[i], dftQuery[i]);
      }
    }

    return newDistance;
  }

  public double getDistance(double d, double sax) {
    double value = (d - sax);
    return 2 * value * value;
  }
  
  public void checkIndex() {
    testInvariant(this.root);
  }

  private void testInvariant(SFANode node) {
    if (node.type == NodeType.Leaf) {
      if (node.elements.size() == 0) {
        throw new RuntimeException("Leaf Node has no Elements!");
      } else if (node.children != null && !node.getChildren().isEmpty()) {
        throw new RuntimeException("Leaf Node has Children!");
      }

    } else if (node.type == NodeType.Internal) {
      if (node.elements != null && node.elements.size() != 0) {
        throw new RuntimeException("Internal Node has Elements!");
      } else if (node.children == null || node.getChildren().isEmpty()) {
        throw new RuntimeException("Internal Node has no Children!");
      }
    }
  }

  public void setMinimalHeight(int minimalHeight) {
    this.minimalHeight = minimalHeight;
  }

  @Override
  public boolean equals(Object treeObject) {
    SFATrie tree = (SFATrie) treeObject;
    HashSet<SFANode> thisTree = new HashSet<SFANode>();
    HashMap<SFANode, SFANode> otherTree = new HashMap<SFANode, SFANode>();
    thisTree.add(this.root);
    otherTree.put(tree.root, tree.root);

    while (!thisTree.isEmpty() && !otherTree.isEmpty()) {
      SFANode firstNode = thisTree.iterator().next();
      thisTree.remove(firstNode);
      SFANode firstNode2 = otherTree.remove(firstNode);

      if (firstNode2 != null) {
        // equal? progress in the tree
        if (firstNode.equals(firstNode2)) {
          for (SFANode node : firstNode.getChildren()) {
            thisTree.add(node);
          }

          for (SFANode node : firstNode2.getChildren()) {
            otherTree.put(node, node);
          }
        }
        // not equal
        else {
          firstNode.toString();
          firstNode2.toString();
          return false;
        }
      } else {
        return false;
      }
    }
    return thisTree.isEmpty() && otherTree.isEmpty();
  }


  /**
   * reset IO-costs to 0
   * @param ioCost
   */
  public void resetIoCosts() {
    this.ioBlockRead = 0;
    this.ioTimeSeriesRead = 0;
    this.timeSeriesRead = 0;
  }

  /**
   * add costs for reading a node
   * @param ioCost
   */
  public void addToBlockRead(int blockCost) {
    this.ioBlockRead += (blockCost);
  }

  /**
   * add I-O costs for reading a lead
   * @param ioCost
   */
  public void addToIOTimeSeriesRead(int ioCost) {
    this.ioTimeSeriesRead += (ioCost);
  }

  /**
   * add costs for distance calculations
   * @param timeSeries
   */
  public void addToTimeSeriesRead(int timeSeries) {
    this.timeSeriesRead += (timeSeries);
  }


  /**
   * the costs for reading a node
   * @return
   */
  public long getBlockRead() {
    return this.ioBlockRead;
  }

  /**
   * The costs for reading a leaf
   * @return
   */
  public long getIoTimeSeriesRead() {
    return this.ioTimeSeriesRead;
  }

  /**
   * the costs for distance calculations
   * @return
   */
  public long getTimeSeriesRead() {
    return this.timeSeriesRead;
  }
  
  protected <E> E removeFromMultiMap(TreeMap<Double, List<E>> queue, double distance) {
    List<E> elements = queue.get(distance);
    E top = elements.remove(0);
    if (elements.size()==0) {
      queue.remove(distance);
    }
    return top;
  }

  protected <E> void addToMultiMap(TreeMap<Double, List<E>> queue, E object, double key) {
    List<E> element = queue.get(key);
    if (element == null) {
      queue.put(key, new LinkedList<E>());
    }
    queue.get(key).add(object);
  }


  public boolean writeToDisk(File path) {
    try (ObjectOutputStream out = new ObjectOutputStream(
        new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(path))))){
      out.writeObject(this);
      return true;
    } catch (IOException e) {
      e.printStackTrace();
    }
    return false;
  }

  public static SFATrie loadFromDisk(File path) {
    try (ObjectInputStream in = new ObjectInputStream(
        new BufferedInputStream(new GZIPInputStream(new FileInputStream(path))))) {
      return (SFATrie) in.readObject();
    }
    catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }

  private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
    in.defaultReadObject();
    resetIoCosts();
  }
  
  static public class TimeSeriesWindow implements Serializable {
    private static final long serialVersionUID = -6192378071620042008L;

    byte[] word;
    double[] fourierValues;
    
    int pos;

    public TimeSeriesWindow(
        double[] fourierValues,
        byte[] word,
        int pos) {
      this.word = word;
      this.pos = pos;
      this.fourierValues = fourierValues;
    }
  }


  public class SFANode implements Serializable {
    private static final long serialVersionUID = -645698847993760867L;

//    protected long[] uuid = new long[2];

    // Children
    private SFANode[] children;

    // TODO - extract approximations from pool and store only offsets.
    //      - create a pool of approximations in the Tree and link to it.    
    // the elements for a leaf-node
    private List<TimeSeriesWindow> elements;

    // path to the leaf node
    protected byte[] word;

    // bounding box  
    protected double[] minValues;
    protected double[] maxValues;

    protected NodeType type = NodeType.Internal;

    public SFANode(byte[] word, int symbols, int length) {
      this.type = NodeType.Leaf;
      this.word = word;

//      // generate uuid for leaf node file names
//      if (this.type == NodeType.Leaf) {      
//        UUID uuidGen = UUID.randomUUID();
//        this.uuid[0] = uuidGen.getLeastSignificantBits();
//        this.uuid[1] = uuidGen.getMostSignificantBits();
//      }

      this.minValues = new double[length];
      this.maxValues = new double[length];
      Arrays.fill(this.minValues, Double.MAX_VALUE);
      Arrays.fill(this.maxValues, Double.MIN_VALUE);

      // create Elements
      this.elements = new ArrayList<>();
    }
//
//    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
//      this.uuid = (long[]) in.readObject();
//      this.children = (Map<byte, SFANode>) in.readObject();
//      this.elements = (List<TimeSeriesWindow>) in.readObject();
//      this.word = (byte[]) in.readObject();
//      this.minValues = (double[]) in.readObject();
//      this.maxValues = (double[]) in.readObject();
//      this.type = (NodeType) in.readObject();
//    }
//
//    private void writeObject(ObjectOutputStream o) throws IOException {
//      o.writeObject(this.uuid);
//      o.writeObject(this.children);
//      o.writeObject(this.elements);
//      o.writeObject(this.word);
//      o.writeObject(this.minValues);
//      o.writeObject(this.maxValues);
//      o.writeObject(this.type);
//    }

    public void removeElements() {
      if (this.elements != null) {
        this.elements = null;
      }
    }

    public void clear() {
      this.elements.clear();
    }

    public void addAll(List<TimeSeriesWindow> elements) {
      this.elements.addAll(elements);
    }

    public SFANode addChild(byte key, int symbols, int dimensionality) {
      byte[] newWord = Arrays.copyOf(this.word, this.word.length + 1);
      newWord[newWord.length - 1] = key;
      return addChild(key, new SFANode(newWord, symbols, dimensionality));
    }

    public SFANode addChild(byte key, SFANode childNode) {
      if (this.children == null) {
        this.children = new SFANode[SFATrie.symbols];
      }

      this.type = NodeType.Internal;
      this.children[key] = childNode;

      return childNode;
    }

    public SFANode getChild(byte key) {
      if (this.children != null) {
        return this.children[key];
      }
      return null;
    }

    public Collection<SFANode> getChildren() {
      if (this.children == null) {
        return new ArrayList<SFATrie.SFANode>();
      }
      // find unique nodes due to compression
      HashSet<SFANode> uniqueNodes = new HashSet<SFATrie.SFANode>();
      for (SFANode nodes : this.children) {
        if (nodes!=null) {
          uniqueNodes.add(nodes);
        }
      }
      return uniqueNodes;
    }

    public void addElement(TimeSeriesWindow element) {
      if (this.type == NodeType.Internal) {
        throw new RuntimeException("Called add Time Series on internal node!");
      }
      this.elements.add(element);
      adaptMinMaxValues(element);
    }

    public void adaptMinMaxValues(TimeSeriesWindow element) {
      adaptMinMaxValues(element.fourierValues, element.fourierValues);
    }

    public void adaptMinMaxValues(SFANode node) {
      adaptMinMaxValues(node.minValues, node.maxValues);
    }

    public void adaptMinMaxValues(double[] minData, double[] maxData) {
      // adapt upper and lower bounds
      for (int i = 0; i < minData.length; i++) {
        this.minValues[i] = Math.min(minData[i], this.minValues[i]);
        this.maxValues[i] = Math.max(maxData[i], this.maxValues[i]);
      }
    }

    public List<TimeSeriesWindow> getElements() {
      return this.elements;
    }

    @Override
    public String toString() {
      StringBuffer output = new StringBuffer();
      output.append(this.type + "\t");
      for (byte c : this.word) {
        output.append("" + (byte) c + " ");
      }
      return output.toString();
    }

    public int getDepth() {
      int height = 0;
      for (SFANode node : getChildren()) {
        height = Math.max(height, node.getDepth() + 1);
      }
      return height;
    }

    public long getLeafCount() {
      long leaves = 0;
      for (SFANode node : getChildren()) {
        if (node.isLeaf()) {
          leaves++;
        } else {
          leaves += node.getLeafCount();
        }
      }
      return leaves;
    }

    public boolean isLeaf() {
      return this.type == NodeType.Leaf;
    }

    /**
     * returns the number of children for each node
     *
     * @param leaves
     * @return
     */
    public ArrayList<Integer> getLeafNodeCounts(ArrayList<Integer> leaves) {
      for (SFANode node : getChildren()) {
        if (node.isLeaf()) {
          leaves.add(node.getSize());
        } else {
          node.getLeafNodeCounts(leaves);
        }
      }
      return leaves;
    }

    /**
     * returns the leaf nodes in the tree
     *
     * @param leaves
     * @return
     */
    public ArrayList<SFANode> getLeafNodes() {
      ArrayList<SFANode> leaves = new ArrayList<SFANode>();
      return getLeafNodes(leaves);
    }

    private ArrayList<SFANode> getLeafNodes(ArrayList<SFANode> leaves) {
      for (SFANode node : getChildren()) {
        if (node.isLeaf()) {
          leaves.add(node);
        } else {
          node.getLeafNodes(leaves);
        }
      }
      return leaves;
    }

    /**
     * Returns the number of nodes in the tree
     *
     * @return
     */
    public int getNodeCount() {
      int nodes = 0;
      for (SFANode node : getChildren()) {
        nodes += 1;
        if (node.type == NodeType.Internal) {
          nodes += node.getNodeCount();
        }
      }
      return nodes;
    }

    /**
     * Returns the total number of elements of the node and all its children
     *
     * @return
     */
    public int getTotalSize() {
      int size = 0;
      for (SFANode node : getChildren()) {
        if (node.isLeaf()) {
          size += node.getSize();
        } else {
          size += node.getTotalSize();
        }
      }
      return size;
    }

    public byte[] getWord() {
      return this.word;
    }

    @Override
    public boolean equals(Object obj) {
      SFANode node = (SFANode) obj;
      return node != null
          && Arrays.equals(this.getWord(), node.getWord())
          && node.type == this.type
          && Arrays.equals(this.maxValues, node.maxValues)
          && Arrays.equals(this.minValues, node.minValues)
          && getSize() == node.getSize();
    }

    @Override
    public int hashCode() {
      return Arrays.hashCode(getWord());
    }

//    protected String getFileName() {
//      StringBuffer name = new StringBuffer("/sfa_leaf/leaf_" + this.uuid[0] + "_" + this.uuid[1]);
//
//      for (byte element2 : this.word) {
//        name.append("_" + (int) element2);
//      }
//      return name.toString() + ".dat";
//    }

    public int getSize() {
      if (this.type == NodeType.Leaf) {
        return this.elements.size();
      } else {
        return getChildren().size();
      }
    }

  }
}
