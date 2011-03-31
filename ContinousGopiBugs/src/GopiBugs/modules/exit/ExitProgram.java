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

package GopiBugs.modules.exit;

import GopiBugs.data.ParameterSet;
import GopiBugs.desktop.Desktop;
import GopiBugs.desktop.GopiBugsMenu;
import GopiBugs.main.GopiBugsCore;
import GopiBugs.main.GopiBugsModule;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;

/**
 *
 * @author scsandra
 */
public class ExitProgram implements GopiBugsModule, ActionListener {
    private Desktop desktop;
    
        
    public void initModule() {

        this.desktop = GopiBugsCore.getDesktop(); 
        desktop.addMenuSeparator(GopiBugsMenu.FILE);
        desktop.addMenuItem(GopiBugsMenu.FILE, "Exit..",
                "Exit program", KeyEvent.VK_E, this, null, null);
    }

    public ParameterSet getParameterSet() {
        return null;
    }

    public void setParameters(ParameterSet parameterValues) {
        
    }

    public void actionPerformed(ActionEvent e) {      
        GopiBugsCore.exitGopiBugs();
    }
    
}
