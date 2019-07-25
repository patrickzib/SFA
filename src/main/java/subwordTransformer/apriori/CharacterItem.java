package subwordTransformer.apriori;

import de.mrapp.apriori.Item;

class CharacterItem implements Item {

  private static final long serialVersionUID = -7757150128073205053L;
  private final short character;
  private final int position;
  private final int alphabetSize;

  public CharacterItem(short character, int position, int alphabetSize) {
    this.character = character;
    this.position = position;
    this.alphabetSize = alphabetSize;
  }

  public short getCharacter() {
    return this.character;
  }

  public int getPosition() {
    return this.position;
  }

  @Override
  public int compareTo(Item item) {
    if (item instanceof CharacterItem) {
      CharacterItem cItem = (CharacterItem) item;
      if (this.position < cItem.position) {
        return -1;
      } else if (this.position > cItem.position) {
        return 1;
      } else if (this.character < cItem.character) {
        return -1;
      } else if (this.character > cItem.character) {
        return 1;
      } else if (this.alphabetSize < cItem.alphabetSize) {
        return -1;
      } else if (this.alphabetSize > cItem.alphabetSize) {
        return 1;
      } else {
        return 0;
      }
    } else {
      return 1;
    }
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof CharacterItem)) {
      return false;
    } else {
      CharacterItem cItem = (CharacterItem) obj;
      return this.character == cItem.character && this.position == cItem.position && this.alphabetSize == cItem.alphabetSize;
    }
  }

  @Override
  public int hashCode() {
    return (int) Math.pow((this.alphabetSize + 1), this.position) * (this.character + 1);
  }

  @Override
  public String toString() {
    return String.valueOf(this.alphabetSize * this.position + this.character);
  }

}
