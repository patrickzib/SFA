package subwordTransformer.apriori;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import de.mrapp.apriori.Filter;
import de.mrapp.apriori.FrequentItemSets;
import de.mrapp.apriori.ItemSet;
import subwordTransformer.SupervisedTransformer;
import subwordTransformer.SupervisedUtils;

/**
 * A transformer that uses the Apriori algorithm to find frequent patterns
 * (character sets). Only works with positional alphabets.
 */
public class SupervisedAprioriTransformer extends SupervisedTransformer<AprioriParameter> {

  private List<FrequentItemSets<CharacterItem>> frequentItemSetsPerClass;
  private List<short[]> dictionary;

  /**
   * @param alphabetSize the alphabet size of the input words
   */
  public SupervisedAprioriTransformer(int alphabetSize) {
    super(alphabetSize, true); // AprioriTransformer only works with positional alphabets (SFA)
  }

  /**
   * @param alphabetSize  the alphabet size of the input words
   * @param fillCharacter the character to be used for wildcards (default: -1)
   */
  public SupervisedAprioriTransformer(int alphabetSize, short fillCharacter) {
    super(alphabetSize, true, fillCharacter);
  }

  @Override
  public int getOutputAlphabetSize() {
    return this.getInputAlphabetSize() + 1;
  }

  @Override
  protected void buildDictionary() {
    frequentItemSetsPerClass = new ArrayList<>();
    for (short[][] classWords : this.getWords()) {
      FrequentItemSets<CharacterItem> frequentItemSets = AprioriUtils.getFrequentItemSets(classWords, 0.1, this.getInputAlphabetSize());
      @SuppressWarnings("rawtypes")
      Filter<ItemSet> filter = Filter.forItemSets().bySize(this.getParameter().getMinSize());
      frequentItemSetsPerClass.add(frequentItemSets.filter(filter));
    }
    this.fillDictionary();
  }

  @Override
  protected void updateDictionary() {
    if (this.getParameter().getMinSize() == this.getOldParameter().getMinSize()) {
      this.fillDictionary();
    } else {
      this.buildDictionary();
    }
  }

  private void fillDictionary() {
    dictionary = new ArrayList<>();
    int numClasses = this.getWords().length;
    double max = Math.sqrt((numClasses - 1) / Math.pow(numClasses, 2));
    Map<List<Short>, double[]> patternSupports = new HashMap<>();
    int i = 0;
    for (FrequentItemSets<CharacterItem> frequentItemSets : frequentItemSetsPerClass) {
      for (ItemSet<CharacterItem> itemSet : frequentItemSets) {
        List<Short> pattern = AprioriUtils.itemSetToPatternList(itemSet, this.getFillCharacter());
        if (!patternSupports.containsKey(pattern)) {
          double[] patternSupport = new double[numClasses];
          Arrays.fill(patternSupport, 0.0);
          patternSupport[i] = itemSet.getSupport();
          patternSupports.put(pattern, patternSupport);
        } else {
          patternSupports.get(pattern)[i] = itemSet.getSupport();
        }
      }
      i++;
    }
    for (Entry<List<Short>, double[]> patternSupport : patternSupports.entrySet()) {
      if (SupervisedUtils.sigma(patternSupport.getValue(), max) >= this.getParameter().getMinSupport()) {
        dictionary.add(AprioriUtils.shortListToArray(patternSupport.getKey()));
      }
    }
  }

  @Override
  public short[][] transform(short[] word) {
    return AprioriUtils.transform(word, dictionary, this.getFillCharacter());
  }

}
