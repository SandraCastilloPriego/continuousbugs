/*
 * Copyright 2010
 * This file is part of XXXXXX.
 * XXXXXX is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 * 
 * XXXXXX is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * XXXXXX; if not, write to the Free Software Foundation, Inc., 51 Franklin St,
 * Fifth Floor, Boston, MA 02110-1301 USA
 */
package GopiBugs.modules.simulation;

import GopiBugs.data.BugDataset;
import GopiBugs.data.PeakListRow;
import GopiBugs.util.Range;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import javax.swing.JTextArea;

/**
 *
 * @author bicha
 */
public class World {

    BugDataset training, validation;
    Cell[][] cells;
    List<Bug> population;
    Random rand;
    int cellsPerSide;
    int numberOfCells;
    int jump = 1;
    int cicleNumber = 0;
    int bugLife;
    JTextArea text;
    List<Result> results;
    int printCount = 0;
    int bugsLimitNumber;

    public World(BugDataset training, BugDataset validation, int cellsPerSide, Range range,
            List<Bug> bugs, int numberOfBugsCopies, int bugLife, JTextArea text,
            List<Result> results, int bugsLimitNumber) {
        this.training = training;
        this.validation = validation;
        this.cellsPerSide = cellsPerSide;
        this.numberOfCells = cellsPerSide * cellsPerSide;
        this.population = new ArrayList<Bug>();
        this.rand = new Random();
        this.bugLife = bugLife;
        this.text = text;
        this.bugsLimitNumber = bugsLimitNumber;



        if (results == null) {
            this.results = new ArrayList<Result>();
        } else {
            this.results = results;
        }

        if (training != null) {
            cells = new Cell[cellsPerSide][cellsPerSide];
            for (int i = 0; i < cellsPerSide; i++) {
                cells[i] = new Cell[cellsPerSide];
                for (int j = 0; j < cells[i].length; j++) {
                    cells[i][j] = new Cell(bugLife);
                    this.setSamplesInCell(training.getAllColumnNames(), cells[i][j], range);
                }
            }

            if (bugs == null) {
                for (int i = 0; i < numberOfBugsCopies; i++) {
                    for (PeakListRow row : training.getRows()) {
                        this.addBug(row);
                    }
                }
            } else {
                this.population = bugs;
                for (Bug bug : this.population) {
                    bug.classify(range);

                }
            }

        }

    }

    private void setSamplesInCell(Vector<String> samplesNames, Cell cell, Range range) {
        int pos = range.getRandom();
        String name = samplesNames.elementAt(pos);
        cell.setParameters(name, range, training.getSampleType(name));
    }

    public synchronized void addMoreBugs() {
        for (PeakListRow row : training.getRows()) {
            this.addBug(row);
        }
    }

    public List<Bug> getBugs() {
        /* List<Bug> clone = new ArrayList<Bug>();
        for(Bug bug : this.population){
        clone.add(bug.clone());
        }

        return clone;*/
        return this.population;
    }

    private void addBug(PeakListRow row) {
        boolean isInside = true;
        int cont = 0;
        while (isInside) {
            int X = this.rand.nextInt(this.cellsPerSide - 1);
            int Y = this.rand.nextInt(this.cellsPerSide - 1);
            Bug bug = new Bug(X, Y, cells[X][Y], row, training, bugLife);
            cells[X][Y].addBug(bug);
            this.population.add(bug);
            isInside = false;
            cont++;
            if (cont > numberOfCells) {
                break;
            }
        }
    }

    public void cicle() {
        movement();
        eat();

        for (Cell[] cellArray : cells) {
            for (Cell cell : cellArray) {
                List<Bug> childs = cell.reproduction();
                if (childs != null) {
                    for (Bug child : childs) {
                        cell.addBug(child);
                        this.population.add(child);
                    }
                }
            }
        }

        death();

        if (population.size() > this.bugsLimitNumber) {
            this.purgeBugs();
        }

        this.cicleNumber++;
        this.printCount++;


    }

    private synchronized void movement() {
        for (Bug bug : population) {
            try {
                int direction = rand.nextInt(8);

                int x = bug.getx();
                int y = bug.gety();

                switch (direction) {
                    case 0:
                        this.setBugPosition(bug, x + jump, y);
                        break;
                    case 1:
                        this.setBugPosition(bug, x, y);
                        break;
                    case 2:
                        this.setBugPosition(bug, x, y + jump);
                        break;
                    case 3:
                        this.setBugPosition(bug, x, y - jump);
                        break;
                    case 4:
                        this.setBugPosition(bug, x + jump, y + jump);
                        break;
                    case 5:
                        this.setBugPosition(bug, x + jump, y - jump);
                        break;
                    case 6:
                        this.setBugPosition(bug, x - jump, y + jump);
                        break;
                    case 7:
                        this.setBugPosition(bug, x - jump, y - jump);
                        break;
                }
            } catch (Exception e) {
            }
        }
    }

    private void setBugPosition(Bug bug, int newx, int newy) {
        if (newx > this.cellsPerSide - 1) {
            newx = 1;
        } else if (newx < 0) {
            newx = 99;
        }
        if (newy > this.cellsPerSide - 1) {
            newy = 1;
        } else if (newy < 0) {
            newy = 99;
        }
        bug.getCell().removeBug(bug);
        bug.setPosition(newx, newy);
        cells[newx][newy].addBug(bug);
        bug.setCell(cells[newx][newy]);

    }

    public int getWorldSize() {
        return this.cellsPerSide;
    }

    private synchronized void eat() {
        for (Bug bug : population) {
            try {
                bug.eat(this.population.size());
            } catch (Exception e) {
            }
        }
    }

    public void purgeBugs() {
        Comparator<Bug> c = new Comparator<Bug>() {

            public int compare(Bug o1, Bug o2) {
                if (o1.getScore() > o2.getScore()) {
                    return 1;
                } else {
                    return -1;
                }
            }
        };

       
        Collections.sort(population, c);
        
        for (int i = this.bugsLimitNumber; i < this.population.size(); i++) {
            population.get(i).kill();
        }
    }

    private synchronized void death() {
        List<Bug> deadBugs = new ArrayList<Bug>();
        for (Bug bug : population) {
            try {
                if (bug.isDead()) {
                    deadBugs.add(bug);
                    this.cells[bug.getx()][bug.gety()].removeBug(bug);
                }
            } catch (Exception e) {
            }
        }
        for (Bug bug : deadBugs) {
            this.population.remove(bug);
        }
    }

    /*  public void saveBugs() throws IOException {

    FileWriter fstream = new FileWriter("/home/bicha/Desktop/output.txt");
    BufferedWriter out = new BufferedWriter(fstream);

    for (Bug bug : this.population) {
    if (bug.getSensitivity() > 0.6 && bug.getSpecificity() > 0.6 && bug.getAge() > 300) {
    out.write(bug.getClassifierType().name());
    out.write("\n");


    for (PeakListRow row : bug.getRows()) {
    out.write(String.valueOf(row.getID()));
    out.write("\n");


    }
    out.write(String.valueOf(bug.getSensitivity()) + " - " + String.valueOf(bug.getSpecificity()) + " - " + bug.getAge());
    out.write("\n");
    out.write("--------------------------------------------------------------");
    out.write("\n");



    }
    }
    out.close();
    fstream.close();
    }*/
    public void printResult(Range range) {        
        this.results.clear();
        Comparator<Result> c = new Comparator<Result>() {

            public int compare(Result o1, Result o2) {
                if (o1.score > o2.score) {
                    return 1;
                } else {
                    return -1;
                }
            }
        };


        int contbug = 0;
        for (Bug bug : this.getBugs()) {

            if (bug.getAge() > 400) {
                Result result = new Result();
                result.Classifier = "Linear regression";
                List<Integer> ids = new ArrayList<Integer>();
                for (PeakListRow row : bug.getRows()) {
                    result.addValue(String.valueOf(row.getID()));
                    ids.add(row.getID());
                }

                  TestBug testing = new TestBug(ids, bug.getClassifierType(), training, validation, bug.getRidge());
                  double value = testing.prediction();
                result.score = bug.getScore();
                result.ridge = bug.getRidge();
                   result.realscore = value;

                this.results.add(result);


                contbug++;
            }

            if(contbug == 10){
                break;
            }
        }

        Collections.sort(results, c);

        contbug = 0;
        String result = range.toString() + " \n";

        for (Result r : results) {
            result += r.toString();
            // contbug++;
        }

        this.text.setText(result);

    }

    public List<Result> getResult() {
        return this.results;
    }

    /*public class Result {

        public String Classifier;
        public List<String> values;
        public double score, realscore;
        public double ridge;
        public int count = 1;

        public Result() {
            this.values = new ArrayList<String>();
        }

        public void addValue(String value) {
            this.values.add(value);
        }

        public List<String> getValues() {
            return this.values;
        }

        public boolean isIt(List<String> values, String classifier) {
            for (String val : values) {
                if (!this.values.contains(val)) {
                    return false;
                }
            }
            if (classifier.equals(Classifier)) {
                return true;
            } else {
                return false;
            }
        }

        public void count() {
            count++;
        }

        @Override
        public String toString() {
            String text = Classifier + " Training - " + score + " - " + realscore + " - " + ridge + " \n";
            for (String val : values) {
                text += val + " - ";
            }
            text += "\n ----------------------------------------------------------\n";
            return text;
        }
    }*/
}
