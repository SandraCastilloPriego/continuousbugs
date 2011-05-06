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
package GopiBugs.modules.simulation.test;


import GopiBugs.data.Parameter;
import GopiBugs.data.ParameterType;
import GopiBugs.data.impl.SimpleParameter;
import GopiBugs.data.impl.SimpleParameterSet;

public class TestParameters extends SimpleParameterSet {	
    
    
    public static final Parameter ids = new SimpleParameter(
            ParameterType.STRING, "IDs:",
            "IDs ", null, null, null, null);

    public static final Parameter ridge = new SimpleParameter(
			ParameterType.DOUBLE, "Ridge",
			"Ridge", "", new Double(0.0),
			new Double(0.0), null);

    public TestParameters() {
        super(new Parameter[]{ridge, ids});
    }
}
