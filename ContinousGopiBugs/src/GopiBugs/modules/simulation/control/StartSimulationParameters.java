/*
 * Copyright 2007-2010 VTT Biotechnology
 * This file is part of GopiBugs.
 *
 * GopiBugs is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * GopiBugs is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * GopiBugs; if not, write to the Free Software Foundation, Inc., 51 Franklin St,
 * Fifth Floor, Boston, MA 02110-1301 USA
 */
package GopiBugs.modules.simulation.control;

import GopiBugs.data.Parameter;
import GopiBugs.data.ParameterType;
import GopiBugs.data.impl.SimpleParameter;
import GopiBugs.data.impl.SimpleParameterSet;

public class StartSimulationParameters extends SimpleParameterSet {

    public static final Parameter iterations = new SimpleParameter(
            ParameterType.INTEGER, "Number of iterations",
            "Introduce the number of iterations", new Integer(1000));
    public static final Parameter worldSize = new SimpleParameter(
            ParameterType.INTEGER, "Size of the world",
            "Introduce the size of each side of the world", new Integer(100));
    public static final Parameter bugLimit = new SimpleParameter(
            ParameterType.INTEGER, "Max number of bugs",
            "Maximum number of bugs living in the world", new Integer(1500));
    public static final Parameter numberOfBugs = new SimpleParameter(
            ParameterType.INTEGER, "Number of copies of bugs",
            "Introduce the number of copies of variables", new Integer(3));
    public static final Parameter bugLife = new SimpleParameter(
            ParameterType.INTEGER, "Life of the Bugs",
            "Minimum number of cicles that a bug can live", new Integer(300));
    public static final Parameter maxScore = new SimpleParameter(
            ParameterType.DOUBLE, "Maximum error",
            "Maximum error for reproduction", new Double(10.0));
    public static final Parameter stoppingCriteria = new SimpleParameter(
            ParameterType.INTEGER, "Stopping criteria (%)",
            "% of variables in living in the world", new Integer(30));
    
    public StartSimulationParameters() {
        super(new Parameter[]{iterations, worldSize, bugLimit, numberOfBugs, bugLife, maxScore, stoppingCriteria});
    }
}
