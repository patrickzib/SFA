// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.index;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.TreeMap;
import java.util.UUID;

import sfa.timeseries.TimeSeries;
import sfa.transformation.SFA;
import sfa.transformation.SFADistance;
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
public class SFATrie {
  protected SFANode root;

  protected int wordLength;
  protected int symbols;
  protected int leafThreshold;

  protected int minimalHeight = -1;

  public transient SFA quantization = null;
  public transient SFADistance distance = null;
  protected boolean compressed = false;

  protected transient long ioBlockRead = 0;
  protected transient long ioTimeSeriesRead = 0;
  protected transient long timeSeriesRead = 0;

  public static enum NodeType { Leaf, Internal };

  /**
   * Create a new SFATrie with dimenionality l and threshold 'leafThreshold'.
   * A leaf will be split, once leafThreshold is exceeded.
   * @param l word length of the SFA transformation
   * @param leafThreshold number of ts in a leaf
   * @param quantization
   */
  public SFATrie(int l, int leafThreshold) {
    this.quantization = new SFA(HistogramType.EQUI_DEPTH);
    this.distance = new SFADistance(this.quantization);

    this.symbols = 8;
    this.wordLength = l;

    this.root = new SFANode(new short[0], this.symbols, this.wordLength);
    this.leafThreshold = leafThreshold;

    resetIoCosts();
  }


  /**
   * Inserts the given time series at the given path.
   *
   * @param prefixPath
   *          the path where to insert the element.
   * @param element
   *          the element to be inserted.
   */
  public void buildIndex(TimeSeries ts, int windowLength) {
    // Train the SFA quantization histogram
    this.quantization.fitWindowing(new TimeSeries[]{ts}, windowLength, this.wordLength, this.symbols, true, true);

    // Transform the time series to SFA words
    double[][] words = this.quantization.transformWindowingDouble(ts, this.wordLength);

    // calculate means and stddev
    int size = (ts.getData().length-windowLength)+1;
    double[] means = new double[size];
    double[] stddevs = new double[size];
    TimeSeries.calcIncreamentalMeanStddev(windowLength, ts, means, stddevs);

    // insert each timeseries window
    for (int offset = 0; offset < words.length; offset++) {
      insert(new TimeSeriesWindow(ts, words[offset], this.quantization.quantization(words[offset]), offset, windowLength, means, stddevs), 0, this.root);
    }

    System.out.println("\tLeaves before path-compression " + getLeafCount());
    System.out.println("\tHeight " + getHeight());
    System.out.println("\tNodes " + getNodeCount());
    compress();

    System.out.println("\tLeaves after path-compression " + getLeafCount());
    System.out.println("\tHeight " + getHeight());
    System.out.println("\tNodes " + getNodeCount());
  }


  private void insert(
      SFANode nodeToInsert,
      short[] path,
      int index,
      SFANode node,
      SFANode parentNode) {

    // adapt min-max-bounds
    node.adaptMinMaxValues(nodeToInsert);

    if (node.type == NodeType.Internal) {
      short key = path[index];
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

    short key = element.word[index];
    SFANode childNode = node.getChild(key);

    if (childNode == null) {
      // add a new child node
      childNode = node.addChild(key, this.symbols, this.wordLength);

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
  public void mergeTrees(SFATrie t) {
    // insert all nodes into the merged tree
    SFATrie tree = t;

    // test if children are not contained
    // insert children of root node
    for (SFANode nodes : tree.root.children) {
      if (nodes != null) {
        insert((SFANode) nodes, ((SFANode) nodes).getWord(), 0, this.root, this.root);
      }
    }
    if (tree.root.getElements() != null && !tree.root.getElements().isEmpty()) {
      throw new RuntimeException("error");
    }
    tree.root = null;
    this.compressed = false;
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
            // transform currentNode's ts
            if (currentNode.getElements() != null) {
//              List<TimeSeriesWindow> elements = new ArrayList<>();
              for (TimeSeriesWindow ts : currentNode.getElements()) {
                ts.fourierValues = null;
              }
//              currentNode.elements.clear();
//              currentNode.addAll(elements);
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
//        count++;
      }

//      SFANode[] children = node.children;
//      node.children = new SFANode[count];
//      for (int i = 0, a = 0; i < node.children.length; i++) {
//        if (node.children[i] != null) {
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
  public SFANode getLeafNode(short[] path) {
    SFANode currentNode = this.root;

    for (short element : path) {
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
      double[] dftQuery, short[] wordQuery, TimeSeries query, int k) {
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
            object.ts,
            query,
            object.means[object.offset],
            object.stddev[object.offset],
            Double.MAX_VALUE,
            object.offset);
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

  protected SFANode getNode(short[] path, double epsilonSquare, double error) {
    SFANode currentNode = this.root;

    for (short element : path) {
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
    short[] wordQuery = quantization.quantization(dftQuery);
    
    return searchKNN(dftQuery, wordQuery, query, k);
  }
  
  public SortedListMap<Double, Integer> searchKNN(
      double[] dftQuery, short[] wordQuery, TimeSeries query, int k) {

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
                object.ts,
                query,
                object.means[object.offset],
                object.stddev[object.offset],
                kthBestDistance,
                object.offset);
            if (distance <= kthBestDistance) {
              result.put(distance, object.offset);
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
      short[] wordQuery,
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
   * setzt die IO-Kosten wieder auf 0
   * @param ioCost
   */
  public void resetIoCosts() {
    this.ioBlockRead = 0;
    this.ioTimeSeriesRead = 0;
    this.timeSeriesRead = 0;
  }

  /**
   * F�rgt den I/O-Kosten f�r das Auslesen von Knoten <code>ioCost</code> hinzu
   * @param ioCost
   */
  public void addToBlockRead(int blockCost) {
    this.ioBlockRead += (blockCost);
  }

  /**
   * F�rgt den I/O-Kosten f�r das Auslesen von Zeitreihen <code>ioCost</code> hinzu
   * @param ioCost
   */
  public void addToIOTimeSeriesRead(int ioCost) {
    this.ioTimeSeriesRead += (ioCost);
  }

  public void addToTimeSeriesRead(int timeSeries) {
    this.timeSeriesRead += (timeSeries);
  }


  /**
   * Die Kosten die ein Lesezugriff auf ein Knotenelement erzeugt.
   * Dabei wird davon ausgegangen, dass das Laden von Knoten/Bl�ttern einen Random-Access ausmacht.
   * @return
   */
  public long getBlockRead() {
    return this.ioBlockRead;
  }

  public long getIoTimeSeriesRead() {
    return this.ioTimeSeriesRead;
  }

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

  class TimeSeriesWindow {
    TimeSeries ts;

    short[] word;
    double[] fourierValues;

    int offset;
    int windowSize;

    transient double[] means;
    transient double[] stddev;

    public TimeSeriesWindow(
        TimeSeries ts,
        double[] fourierValues,
        short[] word,
        int offset,
        int windowSize,
        double[] means,
        double[] stddev) {
      this.ts = ts;
      this.word = word;
      this.offset = offset;
      this.windowSize = windowSize;
      this.fourierValues = fourierValues;
      this.means = means;
      this.stddev = stddev;
    }

//    public TimeSeries getWindow() {
//      return this.ts.getSubsequence(
//          this.offset, this.windowSize, this.means[this.offset], this.stddev[this.offset]); // TODO cache
//    }

    public double[] getFourierTransform() {
      return this.fourierValues;
    }
  }


  public class SFANode {
    protected long[] uuid = new long[2];

    // Children
    private SFANode[] children;

    // the elements for a leaf-node
    private List<TimeSeriesWindow> elements;

    // Schranke mit reduzierter Kardinalität
    protected short[] word;

    // Untere und obere Schranke der vorhandenen Zeitreihen
    protected double[] minValues;
    protected double[] maxValues;

    protected NodeType type = NodeType.Internal;

    public SFANode(short[] word, int symbols, int length) {
      this.type = NodeType.Leaf;
      this.word = word;

      // generate uuid for leaf node file names
      UUID uuidGen = UUID.randomUUID();
      this.uuid[0] = uuidGen.getLeastSignificantBits();
      this.uuid[1] = uuidGen.getMostSignificantBits();

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
//      this.children = (Map<Short, SFANode>) in.readObject();
//      this.elements = (List<TimeSeriesWindow>) in.readObject();
//      this.word = (short[]) in.readObject();
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

    public SFANode addChild(short key, int symbols, int dimensionality) {
      short[] newWord = Arrays.copyOf(this.word, this.word.length + 1);
      newWord[newWord.length - 1] = key;
      return addChild(key, new SFANode(newWord, symbols, dimensionality));
    }

    public SFANode addChild(short key, SFANode childNode) {
      if (this.children == null) {
        this.children = new SFANode[SFATrie.this.symbols];
      }

      this.type = NodeType.Internal;
      this.children[key] = childNode;

      // concurrent modification exception
//      if (this.elements != null) {
//        this.elements.remove();
//        this.elements = null;
//      }

      return childNode;
    }

    public SFANode getChild(short key) {
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
      adaptMinMaxValues(element.getFourierTransform(), element.getFourierTransform());
    }

    public void adaptMinMaxValues(SFANode node) {
      adaptMinMaxValues(node.minValues, node.maxValues);
    }

    public void adaptMinMaxValues(double[] minData, double[] maxData) {
      // Untere und obere Schranke anpassen
      for (int i = 0; i < minData.length; i++) {
        this.minValues[i] = Math.min(minData[i], this.minValues[i]);
        this.maxValues[i] = Math.max(maxData[i], this.maxValues[i]);
      }
    }

//    public boolean adjustMinMaxValues(SymbolicMultiDimObject element) {
//      boolean changed = false;
//      List<Integer> refinePositions = new LinkedList<Integer>();
//      double[] elementData = element.getData();
//
//      for (int i = 0; i < elementData.length; i++) {
//        // does it touch any boundary?
//        if (elementData[i] == this.minValues[i]
//            || elementData[i] == this.maxValues[i]) {
//          refinePositions.add(i);
//          changed = true;
//          break;
//        }
//      }
//
//      // calculate new values for changed positions
//      for (Integer i : refinePositions) {
//        if (this.type == NodeType.Leaf) {
//          for (TimeSeriesWindow ts : getElements()) {
//            double[] realData = ts.getFourierTransform();
//            // Untere und obere Schranke anpassen
//            this.minValues[i] = Math.min(realData[i], this.minValues[i]);
//            this.maxValues[i] = Math.max(realData[i], this.maxValues[i]);
//          }
//        } else if (this.type == NodeType.Internal) {
//          for (SFANode child : getChildren()) {
//            // Untere und obere Schranke anpassen
//            this.minValues[i] = Math.min(child.minValues[i], this.minValues[i]);
//            this.maxValues[i] = Math.max(child.maxValues[i], this.maxValues[i]);
//          }
//        }
//      }
//
//      return changed;
//    }

    public List<TimeSeriesWindow> getElements() {
      return this.elements;
    }

    @Override
    public String toString() {
      StringBuffer output = new StringBuffer();
      output.append(this.type + "\t");
      for (short c : this.word) {
        output.append("" + (short) c + " ");
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

    public short[] getWord() {
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

    protected String getFileName() {
      StringBuffer name = new StringBuffer("/sfa_leaf/leaf_" + this.uuid[0] + "_" + this.uuid[1]);

      for (short element2 : this.word) {
        name.append("_" + (int) element2);
      }
      return name.toString() + ".dat";
    }

    public int getSize() {
      if (this.type == NodeType.Leaf) {
        return this.elements.size();
      } else {
        return getChildren().size();
      }
    }

  }
}
