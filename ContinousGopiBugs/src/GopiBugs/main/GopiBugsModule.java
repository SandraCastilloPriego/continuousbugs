/*
 * Copyright 2007-2010 
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

package GopiBugs.main;

import GopiBugs.data.ParameterSet;


/**
 * This interface represents GopiBugs module.
 *
 * @author Taken from MZmine2
 * http://mzmine.sourceforge.net/
 */
public interface GopiBugsModule {

    /**
     * Initialize this module.
     * 
     */
    public void initModule();

    /**
     * Returns module name 
     * 
     * @return Module name
     */
    public String toString();
    
    /**
     * Returns module's current parameters and their values
     * @return Parameter values as ParameterSet or null if module has no parameters
     */
    public ParameterSet getParameterSet();
    
    
    /**
     * Sets current parameters and their values
     * @param parameterValues New parameter values
     */
    public void setParameters(ParameterSet parameterValues);

}
