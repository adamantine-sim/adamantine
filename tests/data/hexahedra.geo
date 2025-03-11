// 8 vertices of the domain
Point(1) = {0, 0, 0, 1e-2};
Point(2) = {0.1, 0.5, 0, 1e-2};
Point(3) = {1.3, 0.8, 0, 1e-2};
Point(4) = {0.9, 0.2, 0, 1e-2};
Point(5) = {0, 0, 1, 1e-2};
Point(6) = {0.1, 0.5, 1.6, 1e-2};
Point(7) = {1.3, 0.8, 3.1, 1e-2};
Point(8) = {0.9, 0.2, 2.1, 1e-2};

// Bottom face
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Top face
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};

// Front face
Line(9) = {2, 6};
Line(10) = {5, 1};
Curve Loop(3) = {1, 9, -5, 10};
Plane Surface(3) = {3};

// Back face
Line(11) = {3, 7};
Line(12) = {8, 4};
Curve Loop(4) = {-3, 11, 7, 12};
Plane Surface(4) = {4};

// Left face
Curve Loop(5) = {4, -10, -8, 12};
Plane Surface(5) = {5};

// Right face
Curve Loop(6) = {2, 11, -6, -9};
Plane Surface(6) = {6};

// Volume
Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};
