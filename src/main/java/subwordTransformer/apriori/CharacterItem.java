package subwordTransformer.apriori;

import de.mrapp.apriori.Item;

class CharacterItem implements Item {

  private static final long serialVersionUID = -7757150128073205053L;
  private final short character;

  public CharacterItem(short character) {
    this.character = character;
  }

  public short getChar() {
    return this.character;
  }

  @Override
  public int compareTo(Item item) {
    if (item instanceof CharacterItem) {
      CharacterItem cItem = (CharacterItem) item;
      if (this.character < cItem.character) {
        return -1;
      } else if (this.character == cItem.character) {
        return 0;
      } else {
        return 1;
      }
    } else {
      return 2;
    }
  }

  @Override
  public boolean equals(Object obj) {
    return obj instanceof CharacterItem && this.character == ((CharacterItem) obj).character;
  }

  @Override
  public int hashCode() {
    return this.character;
  }

}
