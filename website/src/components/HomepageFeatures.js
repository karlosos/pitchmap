import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Powered by Python and Deep Learning',
    Svg: require('../../static/img/undraw_goal_0v5v.svg').default,
    description: (
      <>
      Project was implemented with Python using a lot of libraries. Mainly I've used <i>OpenCV</i> and <i>CVLib</i>. The machine learning models were created with <i>Tensorflow</i> and <i>scikit learn</i>.
      </>
    ),
  },
  {
    title: 'Not robust',
    Svg: require('../../static/img/undraw_Throw_away_re_x60k.svg').default,
    description: (
      <>
        Pitchmap is not robust system. It's only proof of concept project. It is not adapted to real world tracking.
      </>
    ),
  },
  {
    title: 'Without future',
    Svg: require('../../static/img/undraw_schedule_pnbk.svg').default,
    description: (
      <>
      This project is no longer mantained. I'll probably develop new system solving mistakes from this one.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
