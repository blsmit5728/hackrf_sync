// ========================
// Small Table with Gently Splayed Legs
// ========================

// ---- Core parameters ----
top_size          = 40;    // mm (square tabletop: top_size x top_size)
top_thickness     = 4;     // mm
table_height      = 60;    // mm (underside of top to floor)
leg_angle_deg     = 10;    // degrees outward from vertical (gentle splay)

// ---- Leg geometry ----
leg_inset         = 3;     // mm in from each corner under the top
leg_top_w         = 6;     // mm leg width at top (in leg's local X)
leg_top_t         = 4;     // mm leg thickness at top (in leg's local Y)
leg_bottom_w      = 8;     // mm leg width at bottom
leg_bottom_t      = 6;     // mm leg thickness at bottom

// ---- Foot (optional) ----
add_feet          = true;
foot_len          = 12;    // mm length along leg width direction
foot_wid          = 8;     // mm length along leg thickness direction
foot_thickness    = 2;     // mm

$fn = 64;

// ---- Derived ----
attach_z   = -top_thickness/2;                 // underside of top is Z=attach_z
corner_off = top_size/2 - leg_inset;           // XY attach position per corner
horiz_run  = table_height * tan(leg_angle_deg);// how far the bottom moves outward

// ========================
// Modules
// ========================
module tabletop() {
    // Simple square top; swap with rounded_top() if you want fillets.
    translate([0,0,top_thickness/2])
        cube([top_size, top_size, top_thickness], center=true);
}

// Tapered, outward-splayed leg built with hull() between a small
// top "slice" and a bottom "slice" offset outward.
module leg_at_corner(azimuth_deg, xsign, ysign) {
    // Position at the underside near the corner
    translate([xsign*corner_off, ysign*corner_off, attach_z]) {
        // Align leg local +X toward the corner diagonal
        rotate([0,0,azimuth_deg]) {
            // Tapered strut via hull of two thin rectangular slices
            hull() {
                // Top slice (tiny Z thickness)
                translate([-leg_top_w/2, -leg_top_t/2, -0.1])
                    cube([leg_top_w, leg_top_t, 0.2]);

                // Bottom slice, shifted outward along +X and down by table_height
                translate([horiz_run - leg_bottom_w/2, -leg_bottom_t/2, -table_height-0.1])
                    cube([leg_bottom_w, leg_bottom_t, 0.2]);
            }

            // Optional simple foot pad at the bottom
            if (add_feet) {
                translate([horiz_run - foot_len/2, -foot_wid/2, -table_height - foot_thickness])
                    cube([foot_len, foot_wid, foot_thickness]);
            }
        }
    }
}

// ========================
// Assembly
// ========================
union() {
    difference()
    {
        translate([0,0,-2])tabletop();
        translate([0,0,-50])cylinder(h=100,r1=12.7,r2=12.7);
    }

    // Legs aimed toward the four corners (diagonals)
    leg_at_corner( 45, +1, +1);
    leg_at_corner(135, -1, +1);
    leg_at_corner(225, -1, -1);
    leg_at_corner(315, +1, -1);
}

// ========================
// Optional: rounded top (swap into tabletop() if desired)
// ========================
/*
module rounded_top(r=3) {
    // Minkowski fillet: grows all edges by r, then subtract in Z to keep thickness exact
    minkowski() {
        translate([0,0,top_thickness/2])
            cube([top_size - 2*r, top_size - 2*r, top_thickness - 2*r], center=true);
        cylinder(h=r, r=r, center=false);
    }
}
*/
