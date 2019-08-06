package subwordTransformer.apriori;

import java.util.ArrayList;
import java.util.List;

import de.mrapp.apriori.Filter;
import de.mrapp.apriori.FrequentItemSets;
import de.mrapp.apriori.ItemSet;
import subwordTransformer.UnsupervisedTransformer;

/**
 * A transformer that uses the Apriori algorithm to find frequent patterns
 * (character sets). Only works with positional alphabets.
 */
public class AprioriTransformer extends UnsupervisedTransformer<AprioriParameter> {

  private FrequentItemSets<CharacterItem> frequentItemSets;
  private double currentMinSupport = 1;
  private List<short[]> dictionary;

  /**
   * @param alphabetSize the alphabet size of the input words
   */
  public AprioriTransformer(int alphabetSize) {
    super(alphabetSize, true); // AprioriTransformer only works with positional alphabets (SFA)
  }

  /**
   * @param alphabetSize  the alphabet size of the input words
   * @param fillCharacter the character to be used for wildcards (default: -1)
   */
  public AprioriTransformer(int alphabetSize, short fillCharacter) {
    super(alphabetSize, true, fillCharacter);
  }

  @Override
  public int getOutputAlphabetSize() {
    return this.getInputAlphabetSize() + 1;
  }

  @Override
  protected void buildDictionary() {
    currentMinSupport = this.getParameter().getMinSupport();
    frequentItemSets = AprioriUtils.getFrequentItemSets(this.getWords(), currentMinSupport, this.getInputAlphabetSize());
    this.fillDictionary(frequentItemSets);
  }

  @Override
  protected void updateDictionary() {
    if (this.getParameter().getMinSupport() >= currentMinSupport) {
      @SuppressWarnings("rawtypes")
      Filter<ItemSet> filter = Filter.forItemSets().bySupport(this.getParameter().getMinSupport());
      FrequentItemSets<CharacterItem> filteredFrequentItemSets = frequentItemSets.filter(filter);
      this.fillDictionary(filteredFrequentItemSets);
    } else {
      this.buildDictionary();
    }
  }

  private void fillDictionary(FrequentItemSets<CharacterItem> frequentItemSets) {
    @SuppressWarnings("rawtypes")
    Filter<ItemSet> filter = Filter.forItemSets().bySize(this.getParameter().getMinSize());
    FrequentItemSets<CharacterItem> filteredFrequentItemSets = frequentItemSets.filter(filter);
    dictionary = new ArrayList<>();
    for (ItemSet<CharacterItem> itemSet : filteredFrequentItemSets) {
      dictionary.add(AprioriUtils.itemSetToPattern(itemSet, this.getFillCharacter()));
    }
  }

  @Override
  public short[][] transform(short[] word) {
    return AprioriUtils.transform(word, dictionary, this.getFillCharacter());
  }

}
