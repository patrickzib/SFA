package sfa.classification;

import java.util.ArrayList;
import java.util.List;

public class ShotgunClassifierTest extends AbstractClassifierTest{
    @Override
    protected List<DataSet> getDataSets() {
        List<DataSet> dataSets=new ArrayList<>();
        //variable_length;Shotgun;1.0;0.929
        dataSets.add(new DataSet("variable_length", 1.0, 0.929));
        //Coffee;Shotgun;1.0;1.0
        dataSets.add(new DataSet("Coffee", 1.0, 1.0));
        //Beef;Shotgun;0.633;0.8
        dataSets.add(new DataSet("Beef", 0.633, 0.8));
        //CBF;Shotgun;1.0;0.969
        dataSets.add(new DataSet("CBF", 1.0, 0.969));
        return dataSets;
    }

    @Override
    protected Classifier initClassifier() {
        return new ShotgunClassifier();
    }
}
