import ij.*;
import java.util.*;
import net.imglib2.*;
import net.imglib2.converter.*;
import net.imglib2.neighborsearch.*;
import net.imglib2.img.imageplus.*;
import net.imglib2.type.numeric.integer.*;
import net.imglib2.type.numeric.real.*;
import net.imglib2.view.*;
import net.imglib2.interpolation.randomaccess.*;
import net.imglib2.realtransform.*;

/* parameters */
/* number of particle classes */
static final int nChannels = 3;
/* number of particles per channel */
static final int nParticles = 1500;
/* size of particle */
static final int particleRadius = 5;
/* max radius of a particle's influence on others */ 
static final double particleMaxReach = 0.05;
/* neutral distance of two particles */ 
static final double particleNeutralD = 0.05;
/* force multiplier (speeds things up) */
static final double particleStrength = 100;
/* damping per 1s (1 no damping, 0 full damping) */
static final double damp = 0.05;
/* damping when bouncing at world borders */
static final double bounceDamp = 0.01;
/* jitter amount */
static final float jitter = 0.001;
/* RGB colors for each particle class */
static final int[][] classColors = new int[][]{
    {64, 0, 0},
    {0, 64, 0},
    {0, 0, 64},
    {192, 0, 192},
    {192, 192, 0},
    {0, 192, 192}};
/* number of frames per image in the sequence */
static final int framesPerPicture = 200;
/* image intensity scale */
static double intensityScale = 0.65;
    
/* global properties */
static double dt = 1;
static float spacing = 1;
static double maxX;
static double maxY;
static double dampdt;
static double bounceDampdt;
static ArrayList<ArrayList<Particle>> particleChannels;
static ArrayList<ArrayList<RealRandomAccess<DoubleType>>> pixelAccesses;
static ArrayList<RealRandomAccess<DoubleType>> pixelAccess;
static PShader blur;
static int frame = 0;

class Particle extends RealPoint {
  
  private double dx = 0;
  private double dy = 0;
  private final int clazz;
  
  Particle(final double x, final double y, final int clazz) {
    super(x, y);
    this.clazz = clazz;
  }
  
  void draw() {
    fill(classColors[clazz][0], classColors[clazz][1], classColors[clazz][2], 255);
    circle((float)(spacing * position[0]), (float)(spacing * position[1]), particleRadius);
  }
  
  void push(final double ddx, final double ddy, final double dt) {
    dx += dt * ddx;
    dy += dt * ddy;
  }
  
  void push(final Particle other, final double dt, final double scale) {
    final double diffX = other.position[0] - position[0];
    final double diffY = other.position[1] - position[1];
    final double diff = Math.sqrt(diffX * diffX + diffY * diffY) + 0.0001;
    final double nD = particleNeutralD * scale;
    final double force = -Math.max(0, (1.0 - Math.pow(diff / nD, 0.1)) / nD) * particleStrength; 
    push(force * diffX, force * diffY, dt);
  }
  
  void move(final double dt) {
    position[0] += dt * dx;
    position[1] += dt * dy;
    
    if (position[0] < 0) {
      position[0] = -position[0] * bounceDampdt;
      dx = -dx * bounceDampdt;
    }
    if (position[0] > maxX) {
      position[0] = maxX - (position[0] - maxX) * bounceDampdt;
      dx = -dx * bounceDampdt;
    }
    if (position[1] < 0) {
      position[1] = -position[1] * bounceDampdt;
      dy = -dy * bounceDampdt;
    }
    if (position[1] > maxY) {
      position[1] = maxY - (position[1] - maxY) * bounceDampdt;
      dy = -dy * bounceDampdt;
    }
  }
  
  void damp() {
    dx *= dampdt;
    dy *= dampdt;
  }
  
  void jitter(final float d) {
    dx += randomGaussian() * d;
    dy += randomGaussian() * d;
  }
}
 
void createParticles() {
  particleChannels = new ArrayList<ArrayList<Particle>>();
  final int n = nParticles;
  for (int c = 0; c < nChannels; ++c) {
    final ArrayList<Particle> particles = new ArrayList<Particle>();
    for (int i = 0; i < n; ++i)
      particles.add(new Particle(random(width) / spacing, random(width) / spacing, c));
    particleChannels.add(particles);
  } 
}

static RealRandomAccess<DoubleType> makeChannelAccess(final RandomAccessibleInterval<ARGBType> argb, final int c) {

  final RandomAccessibleInterval<UnsignedByteType> channel = Converters.argbChannel(argb, c);
  final RandomAccessibleInterval<DoubleType> doubleImg = Converters.convert(channel, new RealDoubleConverter(), new DoubleType());
  final RandomAccessible<DoubleType> extended = Views.extendMirrorSingle(doubleImg);
  final RealRandomAccessible<DoubleType> interpolant = Views.interpolate(extended, new NLinearInterpolatorFactory<DoubleType>());
  final Scale2D scale = new Scale2D(1.0 / argb.dimension(0), 1.0 / argb.dimension(1));
  final RealRandomAccessible<DoubleType> scaled = RealViews.affineReal(interpolant, scale);
  return scaled.realRandomAccess();
}

static ArrayList<RealRandomAccess<DoubleType>> makeChannelAccesses(final String path) { //<>//

  final ImagePlus imp = IJ.openImage(path);
  final RandomAccessibleInterval<ARGBType> argb = ImagePlusImgs.from(imp);
  
  final ArrayList<RealRandomAccess<DoubleType>> channels = new ArrayList<RealRandomAccess<DoubleType>>();
  for (int c = 1; c <= nChannels; ++c)
    channels.add(makeChannelAccess(argb, c));
  return channels;
}

static ArrayList<ArrayList<RealRandomAccess<DoubleType>>> loadImageSequence(final String path) {
  
  final String[] files = new File(path).list();
  Arrays.sort(files);

  final ArrayList list = new ArrayList();
  for (final String file : files)
    list.add(makeChannelAccesses(path + "/" + file));
  return list;
}

/**
 * Scale image intensity through an inverse cosine cycle.
 */
static double intensityScale(final int frame) {
  
  return intensityScale * (0.5 - 0.5 * Math.cos(2 * Math.PI * frame / (double)framesPerPicture));
}

/**
 * Pick the current image.
 */
static void updatePixelAccess(final int frame) {
  
  pixelAccess = pixelAccesses.get((frame / framesPerPicture) % pixelAccesses.size());
}

void settings() {
  
  System.setProperty("jogl.disable.openglcore", "true");
  size(512, 512, P2D);
  //fullScreen(P2D);
}

void setup() {
  
  //fullScreen();
  background(0);
  blendMode(ADD);
  frameRate(30);
  noStroke();
  blur = loadShader("blur.glsl");
  spacing = min(width, height);
  dt = 1.0 / frameRate;
  dampdt = Math.pow(damp, dt);
  bounceDampdt = Math.pow(bounceDamp, dt);
  maxX = width / spacing;
  maxY = height / spacing;
  createParticles();
  pixelAccesses = loadImageSequence(sketchPath() + "/data/lab");
}

void draw() {
  
  //clear();
  filter(blur);  
  updatePixelAccess(frame);
  
  //background(255);
  
  final double scale = intensityScale(frame);
  for (int c = 0; c < nChannels; ++c) {
    final ArrayList<Particle> particles = particleChannels.get(c);
    KDTree<Particle> tree = new KDTree<Particle>(particles, particles);
    RadiusNeighborSearchOnKDTree<Particle> search = new RadiusNeighborSearchOnKDTree<Particle>(tree);
    final RealRandomAccess<DoubleType> pa = pixelAccess.get(c);
    for (final Particle particle : particleChannels.get(c)) {
      search.search(particle, particleMaxReach, false);
      for (int i = 0; i < search.numNeighbors(); ++i) {
        final Particle otherParticle = search.getSampler(i).get();
        pa.setPosition(otherParticle);
        particle.push(otherParticle, dt, 1.0 - scale * pa.get().getRealDouble() / 255.0);
      }
    }
    for (final Particle particle : particles) {
      particle.jitter(jitter);
      particle.damp();
      particle.move(dt);
      particle.draw();
    }
  }
  //saveFrame("output/frame-####.png");
  ++frame;
}
